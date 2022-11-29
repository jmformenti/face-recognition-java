package io.github.jmformenti.face.configuration;

import java.io.IOException;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import ai.djl.ModelException;
import io.github.jmformenti.face.model.FaceDetectionModel;
import io.github.jmformenti.face.model.FaceEmbeddingModel;

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
