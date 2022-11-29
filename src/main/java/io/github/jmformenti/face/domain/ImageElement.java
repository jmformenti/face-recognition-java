package io.github.jmformenti.face.domain;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;

public class ImageElement {

	private Image image;
	private Image face;
	private Path originalPath;
	private DetectedObjects detectedFaces;
	private List<double[]> embeddings;

	public ImageElement(Image image) {
		super();
		this.image = image;
		this.embeddings = new ArrayList<>();
	}

	public ImageElement(Image image, Path originalPath) {
		this(image);
		this.originalPath = originalPath;
	}

	public Image getImage() {
		return this.image;
	}

	public Image getFace() {
		return face;
	}

	public void setFace(Image face) {
		this.face = face;
	}

	public Path getOriginalPath() {
		return this.originalPath;
	}

	public DetectedObjects getDetectedFaces() {
		return this.detectedFaces;
	}

	public void setDetectedFaces(DetectedObjects detectedFaces) {
		this.detectedFaces = detectedFaces;
	}

	public List<double[]> getEmbeddings() {
		return this.embeddings;
	}

	public void addEmbedding(double[] embedding) {
		embeddings.add(embedding);
	}

	@Override
	public String toString() {
		return originalPath.toString();
	}
}
