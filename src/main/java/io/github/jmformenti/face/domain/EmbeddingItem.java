package io.github.jmformenti.face.domain;

import java.util.List;

public class EmbeddingItem {

	private String label;
	private List<double[]> embeddings;

	public EmbeddingItem(String label, List<double[]> embeddings) {
		super();
		this.label = label;
		this.embeddings = embeddings;
	}

	public String getLabel() {
		return label;
	}

	public void setLabel(String label) {
		this.label = label;
	}

	public List<double[]> getEmbeddings() {
		return embeddings;
	}

	public void setEmbeddings(List<double[]> embeddings) {
		this.embeddings = embeddings;
	}

}
