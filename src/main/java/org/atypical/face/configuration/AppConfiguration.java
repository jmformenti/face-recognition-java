package org.atypical.face.configuration;

import java.io.IOException;

import org.atypical.face.model.FaceDetectionModel;
import org.atypical.face.model.FaceEmbeddingModel;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import ai.djl.ModelException;

@Configuration
public class AppConfiguration {

	@Bean
	public FaceDetectionModel faceDetectionModel() throws ModelException, IOException {
		FaceDetectionModel faceDetectionModel = new FaceDetectionModel();
		faceDetectionModel.init();
		return faceDetectionModel;
	}
	
	@Bean
	public FaceEmbeddingModel faceEmbeddingModel() throws ModelException, IOException {
		FaceEmbeddingModel faceEmbeddingModel = new FaceEmbeddingModel();
		faceEmbeddingModel.init();
		return faceEmbeddingModel;
	}
}
