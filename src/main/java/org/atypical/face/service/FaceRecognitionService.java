package org.atypical.face.service;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;

import org.atypical.face.domain.EmbeddingsHolder;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;

public interface FaceRecognitionService {

	public EmbeddingsHolder generateEmbeddings(Path basePath, boolean doAugmentation, boolean saveDetectedFaces)
			throws IOException;

	public EmbeddingsHolder loadEmbeddings(Path embeddingsHolderPath) throws FileNotFoundException;

	public void saveEmbeddings(EmbeddingsHolder embeddingsHolder, Path embeddingsHolderPath)
			throws FileNotFoundException;

	public DetectedObjects predict(Image image, EmbeddingsHolder embeddingsHolder);

}
