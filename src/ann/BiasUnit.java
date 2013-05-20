package ann;

public class BiasUnit extends Unit {

	public BiasUnit() {
		output = 1;
	}
	
	@Override
	public double activationFunction(double rawinput) {
		return 1;
	}
	
	@Override
	public void setOutput(double output) {
		System.err.println("Output of a bias unit was reset (silent fail)");
	}

	@Override
	public Unit clone() {
		return new BiasUnit();
	}
	
	@Override
	public int numWeights() {
		return 0;
	}

}
