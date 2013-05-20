package ann;

import java.text.DecimalFormat;
import java.util.Random;

public abstract class Unit {
	static DecimalFormat fmt = new DecimalFormat("0.00");

	double[] weights;
	public double output;
	public double error;
	double[] previousUpdates;

	public Unit() {}
	
	public void init(double[] weights) {
		this.weights = weights;
		previousUpdates = new double[weights.length];
	}

	public static double[] randomWeights(int numWeights) {
		double[] weights = new double[numWeights];
		Random r = new Random();
		for (int i = 0; i < weights.length; i++) {
			weights[i] = r.nextDouble();
		}
		return weights;
	}

	public void updateWeight(double delta, int j, double momentum) {
		double changeThisEpoch = delta + momentum*previousUpdates[j];
		weights[j] += changeThisEpoch;
		previousUpdates[j] = changeThisEpoch;
	}

	@Override
	public String toString() {
		String s = "{";
		for (int i = 0; i < numWeights()-1; i++) {
			s += fmt.format(weights[i]) + ", ";
		}
		if (numWeights() > 0) {
			s += fmt.format(weights[numWeights()-1]);
		}
		return s + "}";
	}

	public void setOutput(double output) {
		this.output = output;
	}

	public void setWeights(double[] inWeights) {
		for (int i = 0; i < numWeights(); i++) {
			weights[i] = inWeights[i];
		}
	}

	public double weight(int i) {
		return weights[i];
	}

	public int numWeights() {
		return weights.length;
	}

	@Override
	public abstract Unit clone();
	
	public abstract double activationFunction(double rawinput);

}