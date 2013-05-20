package ann;

public class OutputLayer extends Layer {

	public OutputLayer(int numRealUnits, Layer previousLayer) {
		int weightsPerUnit = previousLayer.numUnitsInclBias();
		units = new Unit[numRealUnits];
		for (int i = 0; i < numUnitsInclBias(); i++) {
    		units[i] = new OutputUnit(weightsPerUnit);
    	}
	}

	@Override
	public int numRealUnits() {
		return numUnitsInclBias();
	}	
}
