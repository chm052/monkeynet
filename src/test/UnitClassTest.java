package test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import ann.BiasUnit;
import ann.HiddenUnit;
import ann.InputUnit;
import ann.OutputUnit;
import ann.Unit;


public class UnitClassTest {

	@Test
	public void testUnitCreation() throws Exception {
		Unit hu = new HiddenUnit(0);
		assertEquals("Empty unit had weights length != 0",
				hu.numWeights(), 0);

		Unit ou = new OutputUnit(1);
		assertEquals("Non-empty unit had wrong number of weights: was " 
				+ ou.numWeights() + ", should have been 1.",
				ou.numWeights(), 1);

		BiasUnit bu = new BiasUnit();
		assertTrue("Bias unit created with output value != 1",
				Math.abs(bu.output-1) < 0.001);
					
		Unit manyWeights = new HiddenUnit(5);
		assertEquals("Non-empty unit had wrong number of weights: was " 
				+ manyWeights.numWeights() + ", should have been 5.",
				manyWeights.numWeights(), 5);
	
		boolean variation = false;
		double firstWeight = manyWeights.weight(0);
		for (int i = 1; i < manyWeights.numWeights(); i++) {
		    double currWeight = manyWeights.weight(i);
		    assertTrue("Weight was generated < 0",
		    		currWeight >= 0);
		    assertTrue("Weight was generated > 1",
		    		currWeight <= 1);
		    if (firstWeight != currWeight) variation = true;
		}
	
		assertTrue("All weights were the same strength", variation);
	    }

	@Test
	public void testUnitAccessors() {
		Unit hu = new HiddenUnit(3);
		
		double[] weights = {0.4, -0.9, 0.1};
		hu.setWeights(weights);
		
		assertEquals("Unit had incorrect number of weights after reset",
				hu.numWeights(), 3);
		
		for (int i = 0; i < weights.length; i++) {
			assertTrue("Unit had incorrect weight after reset",
					Math.abs(hu.weight(i)-weights[i]) < 0.001);
		}
		
		hu.updateWeight(0.2, 1, 0.0);
		assertTrue("Unit had incorrect weight after update",
				Math.abs(hu.weight(1)- (-0.7)) < 0.001);
		
		hu.setOutput(0.5);
		assertTrue("Unit had incorrect output after reset",
				Math.abs(hu.output - 0.5) < 0.001);
		
		BiasUnit bu = new BiasUnit();
		bu.setOutput(0.5);
		assertTrue("Bias unit had output != 1 after reset",
				Math.abs(bu.output - 1) < 0.001);
	}
	
	@Test
	public void testClone() {
		InputUnit iu = new InputUnit();
		Unit ic = iu.clone();
		
		assertTrue ("Unit clone was instance of the wrong subclass",
				ic instanceof InputUnit);
		assertEquals("Unit clone had wrong number of weights",
				iu.numWeights(), ic.numWeights());
		
		HiddenUnit hu = new HiddenUnit(4);
		Unit hc = hu.clone();
		
		assertTrue ("Unit clone was instance of the wrong subclass",
				hc instanceof HiddenUnit);
		assertEquals("Unit clone had wrong number of weights",
				hu.numWeights(), hc.numWeights());
		
		OutputUnit ou = new OutputUnit(3);
		Unit oc = ou.clone();
		
		assertTrue ("Unit clone was instance of the wrong subclass",
				oc instanceof OutputUnit);
		assertEquals("Unit clone had wrong number of weights",
				ou.numWeights(), oc.numWeights());
	}
}
