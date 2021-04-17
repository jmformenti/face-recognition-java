package org.atypical.face.model;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.atypical.face.domain.ImageElement;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class FaceDetectionModel {

	private ZooModel<Image, DetectedObjects> model;
	private Predictor<Image, DetectedObjects> predictor;

	public void init() throws ModelException, IOException {
		Criteria<Image, DetectedObjects> criteria = Criteria.builder() //
				.optApplication(Application.CV.OBJECT_DETECTION) //
				.setTypes(Image.class, DetectedObjects.class) //
				.optArtifactId("face_detection") //
				.optTranslator(new FaceTranslator(0.5f, 0.7f)) //
				.optFilter("flavor", "server") //
				.build();

		// System.out.println(ModelZoo.listModels());
		this.model = ModelZoo.loadModel(criteria);
		this.predictor = model.newPredictor();
	}

	public void close() {
		this.predictor.close();
		this.model.close();
	}

	private DetectedObjects predict(Image image) throws TranslateException {
		return predictor.predict(image);
	}

	public ImageElement getDetectedFaces(ImageElement imageElement) {
		try {
			DetectedObjects detectedFaces = predict(imageElement.getImage());
			if (detectedFaces.getNumberOfObjects() > 0) {
				imageElement.setDetectedFaces(detectedFaces);
				return imageElement;
			} else {
				return null;
			}
		} catch (TranslateException e) {
			throw new RuntimeException(String.format("Error detecting faces in %s", imageElement.getOriginalPath()), e);
		}
	}

	class FaceTranslator implements Translator<Image, DetectedObjects> {

		private float shrink;
		private float threshold;
		private List<String> className;

		FaceTranslator(float shrink, float threshold) {
			this.shrink = shrink;
			this.threshold = threshold;
			className = Arrays.asList("Not Face", "Face");
		}

		@Override
		public NDList processInput(TranslatorContext ctx, Image input) {
			return processImageInput(ctx.getNDManager(), input, shrink);
		}

		private NDList processImageInput(NDManager manager, Image input, float shrink) {
			NDArray array = input.toNDArray(manager);
			array = NDImageUtils.resize(array, 224, 224);
			array = array.transpose(2, 0, 1).flip(0); // HWC -> CHW BGR -> RGB
			NDArray mean = manager.create(new float[] { 104f, 117f, 123f }, new Shape(3, 1, 1));
			array = array.sub(mean).mul(0.007843f); // normalization
			array = array.expandDims(0); // make batch dimension
			return new NDList(array);
		}

		@Override
		public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
			return processImageOutput(list, className, threshold);
		}

		DetectedObjects processImageOutput(NDList list, List<String> className, float threshold) {
			NDArray result = list.singletonOrThrow();
			float[] probabilities = result.get(":,1").toFloatArray();
			List<String> names = new ArrayList<>();
			List<Double> prob = new ArrayList<>();
			List<BoundingBox> boxes = new ArrayList<>();
			for (int i = 0; i < probabilities.length; i++) {
				if (probabilities[i] >= threshold) {
					float[] array = result.get(i).toFloatArray();
					names.add(className.get((int) array[0]));
					prob.add((double) probabilities[i]);
					boxes.add(new Rectangle(array[2], array[3], array[4] - array[2], array[5] - array[3]));
				}
			}
			return new DetectedObjects(names, prob, boxes);
		}

		@Override
		public Batchifier getBatchifier() {
			return null;
		}
	}
}
