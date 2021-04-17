package org.atypical.face.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

import org.atypical.face.domain.EmbeddingsHolder;
import org.atypical.face.util.ImageUtil;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;

@SpringBootTest
class FaceRecognitionServiceTest {

	private Logger logger = LoggerFactory.getLogger(FaceRecognitionServiceTest.class);

	private static final Path TEST_IMAGES_PATH = Paths.get("src/test/resources/images");

	private static final Path EMBEDDINGS_PATH = Paths.get("target/embeddings.dat");

	@Autowired
	private FaceRecognitionService faceRecognitionService;

	@BeforeAll
	static void beforeAll() throws IOException {
		Files.deleteIfExists(EMBEDDINGS_PATH);
	}

	@Test
	void testGenerateEmbedding() throws IOException {
		EmbeddingsHolder embeddingsHolder = faceRecognitionService.generateEmbeddings(TEST_IMAGES_PATH.resolve("train"),
				false, false);

		assertNotNull(embeddingsHolder);
		assertEquals(13, embeddingsHolder.get("ben_afflek").size());
		assertEquals(17, embeddingsHolder.get("elton_john").size());
		assertEquals(21, embeddingsHolder.get("jerry_seinfeld").size());
		assertEquals(19, embeddingsHolder.get("madonna").size());
		assertEquals(22, embeddingsHolder.get("mindy_kaling").size());

		faceRecognitionService.saveEmbeddings(embeddingsHolder, EMBEDDINGS_PATH);
		assertTrue(Files.exists(EMBEDDINGS_PATH));
	}

	@Test
	void testPredict() throws IOException {
		EmbeddingsHolder embeddingsHolder = faceRecognitionService.loadEmbeddings(EMBEDDINGS_PATH);

		List<Integer> result = Files.walk(TEST_IMAGES_PATH.resolve("val")) //
				.filter(p -> !Files.isDirectory(p)) //
				.map(p -> {
					logger.debug("Predicting {}", p);
					Image image = ImageUtil.getImage(p);
					DetectedObjects faces = faceRecognitionService.predict(image, embeddingsHolder);
					assertTrue(faces.getNumberOfObjects() >= 1);
					DetectedObject maxFace = ImageUtil.selectMaxDetectedObject(faces);
					logger.debug("expected {} label {}", p.getParent().getFileName().toString(),
							maxFace.getClassName());
					if (maxFace.getClassName().equals(p.getParent().getFileName().toString())) {
						return 1;
					} else {
						return 0;
					}
				}).collect(Collectors.toList());

		int ok = result.stream().filter(r -> r == 1).mapToInt(Integer::intValue).sum();
		int total = result.size();
		logger.info("Wrong classified faces: {} / {}", total - ok, total);
		assertEquals(0.96, ok / (double) total, 0.01);
	}
}
