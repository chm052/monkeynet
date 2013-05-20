package ann;

public class InputUnit extends Unit {
	
	public InputUnit() {
		double[] weights = new double[1];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = 1;
		}
		init(weights);
	}

	@Override
	public double activationFunction(double rawinput) {
		return 1.0/(1 + Math.exp((-1) * rawinput));
	}

	@Override
	public Unit clone() {
		return new InputUnit();
	}

}
