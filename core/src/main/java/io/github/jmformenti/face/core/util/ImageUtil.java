package io.github.jmformenti.face.core.util;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import javax.imageio.ImageIO;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Rectangle;
import net.coobird.thumbnailator.Thumbnails;

public class ImageUtil {

	public static final String FACE_IMAGE_TYPE = "jpeg";

	public static boolean isImage(Path path) {
		try {
			return Files.probeContentType(path).startsWith("image/");
		} catch (Exception e) {
			return false;
		}
	}

	/**
	 * Using Thumbnailator library to rotate jpeg files depending the exif info
	 * (https://stackoverflow.com/a/26130136)
	 * 
	 * @param path image input path
	 * @return image prepared
	 */
	public static Image getImage(Path path) {
		try {
			BufferedImage bufferedImage = Thumbnails.of(path.toString()).scale(1).asBufferedImage();

			ByteArrayOutputStream os = new ByteArrayOutputStream();
			ImageIO.write(bufferedImage, "jpeg", os);
			InputStream is = new ByteArrayInputStream(os.toByteArray());

			return ImageFactory.getInstance().fromInputStream(is);
		} catch (IOException e) {
			throw new RuntimeException(String.format("Error reading image %s", path), e);
		}
	}

	public static Image getDetectedObjectImage(Image originalImage, DetectedObject detectedFace) {
		int imageWidth = originalImage.getWidth();
		int imageHeight = originalImage.getHeight();

		Rectangle rectangle = detectedFace.getBoundingBox().getBounds();

		int x = (int) (rectangle.getX() * imageWidth);
		x = x < 0 ? 0 : x;
		int y = (int) (rectangle.getY() * imageHeight);
		y = y < 0 ? 0 : y;
		int w = (int) (rectangle.getWidth() * imageWidth);
		int h = (int) (rectangle.getHeight() * imageHeight);

		w = x + w > imageWidth ? w - (x + w - imageWidth) : w;
		h = y + h > imageHeight ? h - (y + h - imageHeight) : h;

		return originalImage.getSubimage(x, y, w, h);
	}

	public static DetectedObject selectMaxDetectedObject(DetectedObjects detectedFaces) {
		DetectedObjects.DetectedObject maxFaceDetected = null;

		double maxArea = 0;
		List<DetectedObjects.DetectedObject> list = detectedFaces.items();
		for (DetectedObjects.DetectedObject detectedFace : list) {
			Rectangle rect = detectedFace.getBoundingBox().getBounds();
			double area = rect.getHeight() * rect.getWidth();
			if (area > maxArea) {
				maxFaceDetected = detectedFace;
				maxArea = area;
			}
		}

		return new DetectedObject(maxFaceDetected.getClassName(), maxFaceDetected.getProbability(),
				maxFaceDetected.getBoundingBox());
	}

}
