package io.github.jmformenti.face.domain;

public class EmbeddingResult {

	private String name;
	private double probability;

	public EmbeddingResult(String name, double probability) {
		super();
		this.name = name;
		this.probability = probability;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public double getProbability() {
		return probability;
	}

	public void setProbability(double probability) {
		this.probability = probability;
	}

}
