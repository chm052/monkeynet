package ann;

import com.google.common.annotations.VisibleForTesting;

public abstract class Layer {

	@VisibleForTesting
	public Unit[] units;

	public abstract int numRealUnits();

	@VisibleForTesting
	public void initWithModelUnit(Unit model) {
		for (int i = 0; i < numRealUnits(); i++) {
			units[i] = model.clone();
		}
		if (!(model instanceof OutputUnit)) {
			units[numRealUnits()] = new BiasUnit();
		}
	}

	public int numUnitsInclBias() {
		return units.length;
	}

	@Override
	public String toString() {
		String s = "\n";
		for (int i = 0; i < numUnitsInclBias()-1; i++) {
			s += unit(i) + ", ";
		}
		return s + units[numUnitsInclBias()-1];
	}

	public void setOutput(Pattern pattern) {
		for (int i = 0; i < numRealUnits(); i++) {
			unit(i).setOutput(pattern.input(i));
		}
	}

	public void propagate(Layer other) {
		for (int i = 0; i < numRealUnits(); i++) {
			double summedinput = 0;
			for (int j = 0; j < other.numUnitsInclBias(); j++) {
				summedinput += other.unit(j).output * unit(i).weight(j);
			}
			unit(i).output = unit(i).activationFunction(summedinput);
		}
	}

	public static double activationFunction(double rawinput) {
		return 1.0/(1 + Math.exp((-1) * rawinput));
	}

	public double[] getOutput() {
		double[] output = new double[numRealUnits()]; // ignore bias
		for (int i = 0; i < numRealUnits(); i++) {
			output[i] = unit(i).output;
		}
		return output;
	}

	public void setOuterErrors(Pattern target) {
		for (int i = 0; i < numRealUnits(); i++) {
			double out = unit(i).output;
			double daf = out * (1 - out); // derivative of the activation function
			unit(i).error = (target.output(i) - out) * daf;
		}
	}

	// TODO exclude biases from having error assigned to them
	public void setHiddenErrors(Layer next) {
		for (int i = 0; i < numUnitsInclBias(); i++) {
			double error = 0;
			for (int j = 0; j < next.numUnitsInclBias(); j++) {
				error += next.unit(j).error * next.unit(j).weight(i);
			}
			if (!(unit(i) instanceof BiasUnit)) {
				error *= unit(i).output * (1 - unit(i).output);
			}
			unit(i).error = error;
			if (unit(i) instanceof BiasUnit) {
				//System.err.println(error);
			}
		}
	}

	public void updateWeights(Layer previous, double learningSpeed, double momentum) {
		for (int i = 0; i < numRealUnits(); i++) {
			for (int j = 0; j < previous.numUnitsInclBias(); j++) {
				double prevout = previous.unit(j).output;
				double delta = learningSpeed * unit(i).error * prevout;
				unit(i).updateWeight(delta, j, momentum);
			}
		}
	}

	public Unit unit(int i) {
		return units[i];
	}

}