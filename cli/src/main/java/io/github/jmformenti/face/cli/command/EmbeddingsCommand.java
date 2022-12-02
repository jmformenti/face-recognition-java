package io.github.jmformenti.face.cli.command;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.Callable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import ai.djl.engine.Engine;
import io.github.jmformenti.face.core.domain.EmbeddingsHolder;
import io.github.jmformenti.face.core.service.FaceRecognitionService;
import picocli.CommandLine.Command;
import picocli.CommandLine.ExitCode;
import picocli.CommandLine.Option;

@Component
@Command(name = "embed", description = "generates face embeddings.", mixinStandardHelpOptions = true, exitCodeOnExecutionException = 1)
public class EmbeddingsCommand implements Callable<Integer> {

	private Logger logger = LoggerFactory.getLogger(EmbeddingsCommand.class);

	@Autowired
	private FaceRecognitionService faceRecognitionService;

	@Option(names = { "-p", "--path" }, description = "base path with people images.", required = true)
	private String baseDir;

	@Option(names = { "-e", "--epath" }, description = "embeddings file path.", required = true)
	private String embeddingModelPath;

	@Option(names = { "-a", "--doaug" }, description = "do data augmentation for faces.", defaultValue = "false")
	private Boolean doAugmentation;

	@Option(names = { "-s", "--save-faces" }, description = "save detected faces.", defaultValue = "false")
	private Boolean saveDetectedFaces;

	@Override
	public Integer call() throws Exception {
		if (logger.isDebugEnabled()) {
			Engine.debugEnvironment();
		}

		Path basePath = Paths.get(baseDir);
		if (Files.exists(basePath)) {
			logger.info("Analyzing {} dir ..", basePath);
			EmbeddingsHolder embeddingModel = faceRecognitionService.generateEmbeddings(basePath, doAugmentation,
					saveDetectedFaces);

			logger.info("Saving embedding ..");
			faceRecognitionService.saveEmbeddings(embeddingModel, Paths.get(embeddingModelPath));

			logger.info("done.");
			return ExitCode.OK;
		} else {
			logger.error("Input path {} not exists.", basePath);
			return ExitCode.SOFTWARE;
		}
	}
}
