package io.github.jmformenti.face.model;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.compress.utils.IOUtils;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import io.github.jmformenti.face.domain.EmbeddingResult;
import io.github.jmformenti.face.domain.EmbeddingsHolder;
import io.github.jmformenti.face.domain.ImageElement;
import io.github.jmformenti.face.util.ImageUtil;

public class FaceEmbeddingModel {

	private static final double SIMILARITY_THRESHOLD = 0.5;

	private ZooModel<Image, double[]> model;
	private Predictor<Image, double[]> predictor;
	private Kryo kryo;

	public void init() throws ModelException, IOException {
		Resource resource = new ClassPathResource("models/pytorch/vggface2/vggface2.pt");

		File tempModelDir = extractModeltoTempDir(resource).toFile();

		Criteria<Image, double[]> criteria = Criteria.builder() //
				.setTypes(Image.class, double[].class) //
				.optTranslator(new FaceTranslator()) //
				.optEngine("PyTorch") //
				.optModelUrls(tempModelDir.toURI().toString()) //
				.optModelName("vggface2") //
				.build();

		this.model = ModelZoo.loadModel(criteria);
		this.predictor = model.newPredictor();

		deleteFolderOnExit(tempModelDir);

		initKryo();
	}

	private Path extractModeltoTempDir(Resource resource) throws IOException {
		Path tempModelDir = Files.createTempDirectory(resource.getFilename());

		IOUtils.copy(resource.getInputStream(),
				new FileOutputStream(tempModelDir.resolve(resource.getFilename()).toFile()));
		return tempModelDir;
	}

	public void close() {
		this.predictor.close();
	}

	private void initKryo() {
		this.kryo = new Kryo();
		this.kryo.register(EmbeddingsHolder.class);
		this.kryo.register(HashMap.class);
		this.kryo.register(ArrayList.class);
		this.kryo.register(double[].class);
		this.kryo.register(float[].class);
	}

	public double[] predict(Image image) throws TranslateException {
		return predictor.predict(image);
	}

	public ImageElement calculateEmbeddings(ImageElement imageElement) {
		List<DetectedObject> detectedFaces = imageElement.getDetectedFaces().items().stream()
				.map(i -> (DetectedObject) i).collect(Collectors.toList());

		for (DetectedObject detectedFace : detectedFaces) {
			Image face = ImageUtil.getDetectedObjectImage(imageElement.getImage(), detectedFace);
			try {
				imageElement.addEmbedding(predict(face));
			} catch (TranslateException e) {
				throw new RuntimeException(
						String.format("Error generating embedding faces in %s", imageElement.getOriginalPath()), e);
			}
		}
		return imageElement;
	}

	public EmbeddingResult predict(double[] embedding, EmbeddingsHolder embeddingModel) {
		String result = null;
		double minRatioSimilarity = 1;

		try (NDManager manager = NDManager.newBaseManager()) {
			NDArray embeddingToCheck = manager.create(embedding);

			for (String label : embeddingModel.getLabels()) {
				int numVotes = 0;
				double accumulatedSimilarity = 0;

				// TODO do multi-threading search
				for (double[] em : embeddingModel.get(label)) {
					NDArray emArr = manager.create(em);
					double similarity = cosine(emArr, embeddingToCheck);
					if (similarity <= SIMILARITY_THRESHOLD) {
						accumulatedSimilarity += similarity;
						numVotes++;
					}
				}

				double ratioSimilarity = accumulatedSimilarity / numVotes;
				if (ratioSimilarity < minRatioSimilarity) {
					result = label;
					minRatioSimilarity = ratioSimilarity;
				}
			}
		}

		if (result == null) {
			return null;
		} else {
			return new EmbeddingResult(result, 1 - minRatioSimilarity);
		}
	}

	public void save(EmbeddingsHolder embeddingModel, Path embeddingModelPath) throws FileNotFoundException {
		Output output = new Output(new FileOutputStream(embeddingModelPath.toFile()));
		kryo.writeClassAndObject(output, embeddingModel);
		output.close();
	}

	public EmbeddingsHolder read(Path embeddingModelPath) throws FileNotFoundException {
		Input input = new Input(new FileInputStream(embeddingModelPath.toFile()));
		Object embeddingModel = kryo.readClassAndObject(input);
		input.close();
		return (EmbeddingsHolder) embeddingModel;
	}

	// Translated from scipy.spatial.distance.cosine
	public static double cosine(NDArray u, NDArray v) {
		double uv = u.mul(v).mean().toDoubleArray()[0];
		NDArray uu = u.pow(2).mean();
		NDArray vv = v.pow(2).mean();
		return Math.abs(1 - uv / uu.mul(vv).sqrt().toDoubleArray()[0]);
	}

	private void deleteFolderOnExit(File folder) {
		folder.deleteOnExit();
		for (File f : folder.listFiles()) {
			f.deleteOnExit();
		}
	}

	class FaceTranslator implements Translator<Image, double[]> {

		public FaceTranslator() {
		}

		@Override
		public double[] processOutput(TranslatorContext ctx, NDList list) {
			if (list != null && !list.isEmpty()) {
				float[] floatArray = list.get(0).toFloatArray();
				return IntStream.range(0, floatArray.length).mapToDouble(i -> floatArray[i]).toArray();
			} else {
				return null;
			}
		}

		@Override
		public NDList processInput(TranslatorContext ctx, Image input) {
			NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);

			Resize resize = new Resize(160, 160);
			array = resize.transform(array);

			// fixed image standardization (used in MTCNN post process faces for trained
			// pytorch model)
			array = array.sub(0.498).div(0.5);

			array = array.expandDims(0);

			array = array.getNDArrayInternal().toTensor();
			return new NDList(array);
		}

		@Override
		public Batchifier getBatchifier() {
			return null;
		}
	}

}
