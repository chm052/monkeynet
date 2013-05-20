package test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import ann.ANN;
import ann.InputLayer;
import ann.Layer;
import ann.OutputLayer;
import ann.Pattern;

public class ANNTest {

	@Test
	public void testANNCreation() {

		int[] ls1 = {1, 1};
		ANN ann = new ANN(0, 0.5, 0.1, ls1[0], ls1[1]);
		double maxError = 0.5;

		assertTrue("Wrong learning speed",
				Math.abs(ann.learningSpeed-0.5) < 0.001);
		assertTrue("Wrong momentum",
				Math.abs(ann.momentum-0.1) < 0.001);
		assertEquals("Wrong number of layers",
				ann.numLayers(), 2);
		for (int i = 0; i < ann.numLayers(); i++) {
			assertEquals("Layer " + i + " had wrong number of units",
					ann.layerSize(i), ls1[i]);
		}
		
		int[] ls2 = {2, 5, 1, 6};
		ann = new ANN(2, 0.0, 0.95, ls2[0], ls2[1], 
				ls2[2], ls2[3]);
		
		assertEquals("Wrong number of layers",
				ann.numLayers(), 4);
		for (int i = 0; i < ann.numLayers(); i++) {
			assertEquals("Layer " + i + " had wrong number of units",
					ann.layerSize(i), ls2[i]);
		}
	}

	@Test
	public void testSetAsInput() {
		ANN ann = new ANN(2, 0.1, 0.1, 
				4, 3, 2, 1);
		double[] in = {0.5, 0.65, -0.1, 0.2};
		ann.setAsInput(new Pattern(in, null));
		
		assertTrue("ANN input was wrong. Was: "
				+ Arrays.toString(ann.getOutput(0))
				+ ", should have been: "
				+ Arrays.toString(in),
				Arrays.equals(in, ann.getOutput(0)));
	}
	
	@Test
	public void testPass() {
		ANN ann = new ANN(0, 0.5, 1.0, 1, 1);
		
		Layer in = new InputLayer(1);
		
		double[] weights = {0.6, -0.2};
		Layer out = new OutputLayer(1, in);
		out.unit(0).setWeights(weights);
		
		ann.setLayer(in, 0);
		ann.setLayer(out, 1);
		
		double[] input = {0.5};
		double[] expectedOutput = {0.5249791874789399};
		
		ann.pass(new Pattern(input, null));
		
		double[] output = ann.getCurrentOutput();
		assertTrue("Propagation gave the wrong output. Was: " + Arrays.toString(output)
				+ ", should have been: " + Arrays.toString(expectedOutput),
				Arrays.equals(output, expectedOutput));
	}
	
}
