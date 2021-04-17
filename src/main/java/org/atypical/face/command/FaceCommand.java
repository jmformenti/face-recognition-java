package org.atypical.face.command;

import java.util.concurrent.Callable;

import org.springframework.stereotype.Component;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.ExitCode;

@Component
@Command(name = "face", mixinStandardHelpOptions = true, versionProvider = FaceVersion.class, subcommands = {
		EmbeddingsCommand.class, PredictCommand.class })
public class FaceCommand implements Callable<Integer> {

	@Override
	public Integer call() throws Exception {
		new CommandLine(new FaceCommand()).usage(System.out);
		return ExitCode.OK;
	}
}
