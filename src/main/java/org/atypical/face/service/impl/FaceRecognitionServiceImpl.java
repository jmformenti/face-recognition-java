package org.atypical.face.service.impl;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.compress.utils.FileNameUtils;
import org.atypical.face.domain.EmbeddingItem;
import org.atypical.face.domain.EmbeddingResult;
import org.atypical.face.domain.EmbeddingsHolder;
import org.atypical.face.domain.ImageElement;
import org.atypical.face.model.FaceDetectionModel;
import org.atypical.face.model.FaceEmbeddingModel;
import org.atypical.face.service.FaceRecognitionService;
import org.atypical.face.util.ImageUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;

@Service
public class FaceRecognitionServiceImpl implements FaceRecognitionService {

	private Logger logger = LoggerFactory.getLogger(FaceRecognitionServiceImpl.class);

	private static final String UNKNOWN_LABEL = "unknown";

	private static final String FACE_FOLDER = "face";

	@Autowired
	private FaceDetectionModel faceDetection;

	@Autowired
	private FaceEmbeddingModel faceEmbedding;

	@Override
	public EmbeddingsHolder generateEmbeddings(Path basePath, boolean doAugmentation, boolean saveDetectedFaces)
			throws IOException {
		EmbeddingsHolder embeddingModel = new EmbeddingsHolder();

		Files.walk(basePath) //
				.filter(p -> Files.isDirectory(p) && !p.equals(basePath)) //
				.map(p -> getFacesEmbedding(p, doAugmentation, saveDetectedFaces)) //
				.forEach(e -> embeddingModel.add(e));

		return embeddingModel;
	}

	private EmbeddingItem getFacesEmbedding(Path personPath, boolean doAugmentation, boolean saveDetectedFaces) {
		logger.debug("Analyzing {} dir ..", personPath);
		try {
			List<double[]> embeddings = Files.walk(personPath) //
					.filter(path -> ImageUtil.isImage(path)) //
					.map(path -> {
						logger.debug("Found image {}", path);
						return new ImageElement(ImageUtil.getImage(path), path);
					}) //
					.map(imageElem -> detectFaces(imageElem, saveDetectedFaces)) //
					.map(imageElem -> faceEmbedding.calculateEmbeddings(imageElem)) //
					.map(imageElem -> generateDataAugmentation(imageElem, doAugmentation)) //
					.flatMap(imageElem -> imageElem.getEmbeddings().stream()) //
					.collect(Collectors.toList());

			return new EmbeddingItem(getLabel(personPath), embeddings);
		} catch (IOException e) {
			throw new RuntimeException(String.format("Error analyzing images from %s", personPath), e);
		}
	}

	private ImageElement detectFaces(ImageElement imageElement, boolean saveDetectedFaces) {
		imageElement = faceDetection.getDetectedFaces(imageElement);
		if (imageElement.getDetectedFaces().getNumberOfObjects() > 0) {
			logger.debug("Found {} faces.", imageElement.getDetectedFaces().getNumberOfObjects());
			if (imageElement.getDetectedFaces().getNumberOfObjects() > 1) {
				logger.debug("Selecting max face to use.");
				DetectedObject maxFace = ImageUtil.selectMaxDetectedObject(imageElement.getDetectedFaces());
				imageElement.setDetectedFaces(new DetectedObjects(List.of(maxFace.getClassName()),
						List.of(maxFace.getProbability()), List.of(maxFace.getBoundingBox())));
			}
			if (saveDetectedFaces) {
				saveDetectedFaces(imageElement, imageElement.getDetectedFaces());
			}
			return imageElement;
		} else {
			return null;
		}
	}

	private void saveDetectedFaces(ImageElement imageElement, DetectedObjects detectedFaces) {
		List<DetectedObject> list = detectedFaces.items();
		int i = 1;
		for (DetectedObject detectedFace : list) {
			Image face = ImageUtil.getDetectedObjectImage(imageElement.getImage(), detectedFace);
			imageElement.setFace(face);
			try {
				face.save(
						Files.newOutputStream(
								getFacePath(imageElement.getOriginalPath(), i, ImageUtil.FACE_IMAGE_TYPE)),
						ImageUtil.FACE_IMAGE_TYPE);
			} catch (IOException e) {
				logger.error(String.format("Error writing face %d for %s", i, imageElement), e);
			}
			i++;
		}
	}

	private ImageElement generateDataAugmentation(ImageElement imageElement, boolean doAugmentation) {
		if (doAugmentation) {
			try (NDManager manager = NDManager.newBaseManager()) {
				IntStream.range(1, 3).forEach(i -> {
					NDArray rotated = NDImageUtils.rotate90(imageElement.getFace().toNDArray(manager), i);
					Image augmentedFace = ImageFactory.getInstance().fromNDArray(rotated);
					try {
						imageElement.getEmbeddings().add(faceEmbedding.predict(augmentedFace));
					} catch (TranslateException e) {
						logger.error(String.format("Error generating augmented face %d for %s", i, imageElement), e);
					}
				});
			}
		}
		return imageElement;
	}

	private Path getFacePath(Path originalImagePath, int numFace, String type) throws IOException {
		Path faceFolder = Paths.get(
				originalImagePath.getParent().getParent().getParent().resolve(FACE_FOLDER).toString(),
				getLabel(originalImagePath.getParent()));
		Files.createDirectories(faceFolder);
		return faceFolder.resolve(
				FileNameUtils.getBaseName(originalImagePath.getFileName().toString()) + "_face" + numFace + "." + type);
	}

	private String getLabel(Path path) {
		return FileNameUtils.getBaseName(path.getFileName().toString());
	}

	@Override
	public EmbeddingsHolder loadEmbeddings(Path embeddingsHolderPath) throws FileNotFoundException {
		return faceEmbedding.read(embeddingsHolderPath);
	}

	@Override
	public void saveEmbeddings(EmbeddingsHolder embeddingsHolder, Path embeddingsHolderPath)
			throws FileNotFoundException {
		faceEmbedding.save(embeddingsHolder, embeddingsHolderPath);
	}

	@Override
	public DetectedObjects predict(Image image, EmbeddingsHolder embeddingsHolder) {
		ImageElement imageElement = new ImageElement(image);
		return recognizeFaces( //
				faceEmbedding.calculateEmbeddings( //
						faceDetection.getDetectedFaces(imageElement)),
				embeddingsHolder);
	}

	private DetectedObjects recognizeFaces(ImageElement imageElement, EmbeddingsHolder embeddingHolder) {
		List<String> names = new ArrayList<>();
		List<Double> prob = new ArrayList<>();
		List<BoundingBox> rect = new ArrayList<>();

		for (int i = 0; i < imageElement.getEmbeddings().size(); i++) {
			EmbeddingResult result = faceEmbedding.predict(imageElement.getEmbeddings().get(i), embeddingHolder);
			if (result == null) {
				names.add(UNKNOWN_LABEL);
				prob.add(0D);
			} else {
				names.add(result.getName());
				prob.add(result.getProbability());
			}
			rect.add(((DetectedObject) imageElement.getDetectedFaces().item(i)).getBoundingBox());
		}

		return new DetectedObjects(names, prob, rect);
	}

}
