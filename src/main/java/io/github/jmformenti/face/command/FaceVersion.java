package io.github.jmformenti.face.command;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import picocli.CommandLine.IVersionProvider;

@Component
public class FaceVersion implements IVersionProvider {

	@Value("${face.version}")
	private String appVersion;

	@Override
	public String[] getVersion() throws Exception {
		return new String[] { "v" + appVersion };
	}
}
