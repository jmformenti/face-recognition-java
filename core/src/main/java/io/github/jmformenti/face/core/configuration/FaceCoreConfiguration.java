package io.github.jmformenti.face.core.configuration;

import java.io.IOException;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import ai.djl.ModelException;
import io.github.jmformenti.face.core.model.FaceDetectionModel;
import io.github.jmformenti.face.core.model.FaceEmbeddingModel;

@Configuration
public class FaceCoreConfiguration {

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
