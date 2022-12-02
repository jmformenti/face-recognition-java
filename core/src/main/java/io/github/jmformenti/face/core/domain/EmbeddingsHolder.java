package io.github.jmformenti.face.core.domain;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class EmbeddingsHolder {

	private Map<String, List<double[]>> embeddingsByLabel;

	public EmbeddingsHolder() {
		super();
		this.embeddingsByLabel = new HashMap<>();
	}

	public void add(EmbeddingItem item) {
		embeddingsByLabel.put(item.getLabel(), item.getEmbeddings());
	}

	public List<double[]> get(String label) {
		return embeddingsByLabel.get(label);
	}

	public Set<String> getLabels() {
		return embeddingsByLabel.keySet();
	}

	public int size() {
		return embeddingsByLabel.size();
	}
}
