package ann;

import java.util.Arrays;
import java.util.Random;

public class Pattern {
	double[] input;
	double[] output;
	double noise = 0.0;
	final Random r = new Random();;

	public Pattern(double[] input, double[] output) {
		this.input = input;
		this.output = output;
	}

	public Pattern(ANN ann) {
		this.input = new double[ann.layerSize(0)];

		for (int i = 0; i < input.length; i++) {
			input[i] = r.nextDouble();
		}

		ann.pass(new Pattern(input, null));
		this.output = ann.getCurrentOutput();
	}
	
	public void setNoise(double noise) {
		this.noise = noise;
	}

	public double input(int i) {
		double addNoise = 0;
		if (r.nextDouble() < noise) {
			addNoise = r.nextDouble();
		}
		return input[i] + addNoise;
	}

	public double output(int i) {
		return output[i];
	}

	@Override
	public String toString() {
		return Arrays.toString(input) + "->" + Arrays.toString(output);
	}
}