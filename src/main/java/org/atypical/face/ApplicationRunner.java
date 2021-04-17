package org.atypical.face;

import org.atypical.face.command.FaceCommand;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.ExitCodeGenerator;
import org.springframework.stereotype.Component;

import picocli.CommandLine;
import picocli.CommandLine.IFactory;

@Component
public class ApplicationRunner implements CommandLineRunner, ExitCodeGenerator {

	private final FaceCommand faceCommand;

	private final IFactory factory; // auto-configured to inject PicocliSpringFactory

	private int exitCode;

	public ApplicationRunner(FaceCommand faceCommand, IFactory factory) {
		this.faceCommand = faceCommand;
		this.factory = factory;
	}

	@Override
	public void run(String... args) throws Exception {
		exitCode = new CommandLine(faceCommand, factory).execute(args);
	}

	@Override
	public int getExitCode() {
		return exitCode;
	}
}
