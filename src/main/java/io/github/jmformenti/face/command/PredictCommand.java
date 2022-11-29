package io.github.jmformenti.face.command;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import org.apache.commons.compress.utils.FileNameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import io.github.jmformenti.face.domain.EmbeddingsHolder;
import io.github.jmformenti.face.service.FaceRecognitionService;
import io.github.jmformenti.face.util.ImageUtil;
import picocli.CommandLine.Command;
import picocli.CommandLine.ExitCode;
import picocli.CommandLine.Option;

@Component
@Command(name = "predict", description = "runs face recognition pipeline.", mixinStandardHelpOptions = true, exitCodeOnExecutionException = 1)
public class PredictCommand implements Callable<Integer> {

	private Logger logger = LoggerFactory.getLogger(PredictCommand.class);

	@Autowired
	private FaceRecognitionService faceRecognitionService;

	@Option(names = { "-p", "--path" }, description = "image path to predict.", required = true)
	private String imageParam;

	@Option(names = { "-e", "--epath" }, description = "embeddings file path.", required = true)
	private String embeddingModelPath;

	@Override
	public Integer call() throws Exception {
		Path imagePath = Paths.get(imageParam);
		if (ImageUtil.isImage(imagePath)) {
			EmbeddingsHolder embeddingsHolder = faceRecognitionService.loadEmbeddings(Paths.get(embeddingModelPath));

			Image image = ImageUtil.getImage(imagePath);
			DetectedObjects faces = faceRecognitionService.predict(image, embeddingsHolder);

			faces = addProbabilityToLabel(faces);
			image.drawBoundingBoxes(faces);
			Path resultPath = getResultPath(imagePath);
			image.save(Files.newOutputStream(resultPath), ImageUtil.FACE_IMAGE_TYPE);

			logger.info("Saved result image in {}", resultPath);
			logger.info("done.");
			return ExitCode.OK;
		} else {
			logger.error("Image path {} not exists.", imagePath);
			return ExitCode.SOFTWARE;
		}
	}

	private DetectedObjects addProbabilityToLabel(DetectedObjects faces) {
		List<String> names = new ArrayList<>();
		List<Double> probs = new ArrayList<>();
		List<BoundingBox> rects = new ArrayList<>();

		List<DetectedObject> allFaces = faces.items();
		int i = 0;
		for (DetectedObject face : allFaces) {
			names.add(String.format("%s - %.2f - %d", face.getClassName(), face.getProbability(), i));
			probs.add(face.getProbability());
			rects.add(face.getBoundingBox());
			i++;
		}

		return new DetectedObjects(names, probs, rects);
	}

	private Path getResultPath(Path originalImagePath) {
		String filenameResult = FileNameUtils.getBaseName(originalImagePath.getFileName().toString()) + "_result.jpg";
		if (originalImagePath.getParent() != null) {
			return originalImagePath.getParent().resolve(filenameResult);
		} else {
			return Paths.get(filenameResult);
		}
	}

}
