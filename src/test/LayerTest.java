package test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import ann.BiasUnit;
import ann.HiddenLayer;
import ann.HiddenUnit;
import ann.InputLayer;
import ann.InputUnit;
import ann.Layer;
import ann.OutputLayer;
import ann.Pattern;
import ann.Unit;

public class LayerTest {

	@Test
	public void testNumUnits() {
		InputLayer il = new InputLayer(3);

		assertEquals("Layer has wrong number of real/total units",
				3, il.numRealUnits());
		assertEquals("Layer has wrong number of real/total units",
				4, il.numUnitsInclBias());

		HiddenLayer hl = new HiddenLayer(4, il);

		assertEquals("Layer has wrong number of real/total units",
				4, hl.numRealUnits());
		assertEquals("Layer has wrong number of real/total units",
				5, hl.numUnitsInclBias());

		OutputLayer ol = new OutputLayer(5, hl);

		assertEquals("Layer has wrong number of real/total units",
				5, ol.numRealUnits());
		assertEquals("Layer has wrong number of real/total units",
				5, ol.numUnitsInclBias());
	}

	@Test
	public void testSetOutput() {
		double[] in = {0.1, 0.2, -0.3};
		double[] out = {0.4, -0.1, 0.05};

		Pattern p = new Pattern(in, out);

		InputLayer il = new InputLayer(3);
		il.setOutput(p);

		for (int i = 0; i < il.numRealUnits(); i++) {
			assertTrue("Output does not match the set pattern",
					Math.abs(il.unit(i).output - in[i]) < 0.001);
		}
	}
	
	@Test
	public void testInitWithModelUnit() {
		int numRealUnits = 3;
		Layer il = new InputLayer(numRealUnits);

		il.units = new Unit[numRealUnits + 1];
		il.initWithModelUnit(new InputUnit());

		assertTrue("Layer units are not of the model class",
				il.unit(0) instanceof InputUnit);
		assertTrue("Last unit is not a bias",
				il.unit(il.numUnitsInclBias()-1) instanceof BiasUnit);

		il.initWithModelUnit(new HiddenUnit(3));
		assertTrue("Layer units are not of the model class",
				il.unit(0) instanceof HiddenUnit);
		assertTrue("Last unit is not a bias",
				il.unit(il.numUnitsInclBias()-1) instanceof BiasUnit);
	}

	@Test
	public void testPropagation() {

		InputLayer inputter = new InputLayer(2);
		HiddenLayer receiver = new HiddenLayer(2, inputter);
		double[][] weights = {{-0.8, 0.7, -0.3}, {0.6, -0.5, 0.4}};
		double[] input = {0.92, 0.06};
		double[] expectedOutput = {0.27, 0.72};
		Pattern p = new Pattern(input, expectedOutput);

		inputter.setOutput(p);

		for (int i = 0; i < receiver.numRealUnits(); i++) {
			receiver.unit(i).setWeights(weights[i]);
		}

		receiver.propagate(inputter);

		double[] actualOutputs = receiver.getOutput();

		for (int i = 0; i < receiver.numRealUnits(); i++) {
			assertTrue("Wrong output of propagation. Was " + actualOutputs[i]
					+ ", should have been " + expectedOutput[i],
					Math.abs(actualOutputs[i]-expectedOutput[i]) < 0.01);
		}

	}

	@Test
	public void testOuterErrors() {
		Layer il = new InputLayer(1);
		Layer ol = new OutputLayer(4, il);

		double[] targetOutputs = {0.9, 0.1, -0.8, 0.55};
		double[] actualOutputs = {0.27, 0.72, 0.05, 0.55};
		double[] errors = {0.124, -0.125, -0.040375, 0.0};

		for (int i = 0; i < ol.numRealUnits(); i++) {
			ol.unit(i).output = actualOutputs[i];
		}

		ol.setOuterErrors(new Pattern(null, targetOutputs));

		for (int i = 0; i < ol.numRealUnits(); i++) {
			assertTrue("Wrong error for unit " + i +
					". Was: " + ol.unit(i).error +
					", should have been: " + errors[i],
					Math.abs(ol.unit(i).error-errors[i]) < 0.001);
		}
	}

	@Test
	public void testHiddenErrors() {

		Layer il = new InputLayer(1);
		Layer hl = new HiddenLayer(2, il);
		Layer ol = new OutputLayer(2, hl);

		double[] outerErrors = {0.124, -0.125};
		double[][] outerWeights = {{-0.3, -0.8, 0.7},
				{0.4, 0.6, -0.5}};

		for (int i = 0; i < ol.numRealUnits(); i++) {
			ol.unit(i).error = outerErrors[i];
			ol.unit(i).setWeights(outerWeights[i]);
		}

		double[] hiddenOutput = {0.92, 0.06};
		for (int i = 0; i < hl.numRealUnits(); i++) {
			hl.unit(i).output = hiddenOutput[i];
		}

		hl.setHiddenErrors(ol);

		double[] hiddenErrors = {-0.013, 0.008};

	}

	@Test
	public void testActivationFunction() {
		double[] inputs = {0, 1, -1};
		double[] outputs = {0.5, 0.73106, 0.26894};
		
		for (int i = 0; i < inputs.length; i++) {
		assertTrue("Wrong output for "
				+ inputs[i] + ". Was: " + 
				Layer.activationFunction(inputs[i]) + 
				", should have been: " + outputs[i],
				Math.abs(Layer.activationFunction(inputs[i])
						- outputs[i]) < 0.01);
		}
	}
}
