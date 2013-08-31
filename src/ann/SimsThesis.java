package ann;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class SimsThesis {
	
	static double[][] inputsIris;
	static double[][] outputsIris;
	static double[][] shortinputsIris;
	static double[][] shortoutputsIris;
	static double[][] inpseudoIris;
	static double[][] outpseudoIris;

	public static void main(String[] args) throws IOException {
		inputsIris = ANN.readPattern("src/ann/irisinputs.txt", "\\s\\s");
		outputsIris = ANN.readPattern("src/ann/irisoutputs.txt", "\\s\\s");
		shortinputsIris = Arrays.copyOf(inputsIris, 30);
		shortoutputsIris = Arrays.copyOf(outputsIris, 30);
		inpseudoIris = Arrays.copyOfRange(inputsIris, 50, 70);
		outpseudoIris = Arrays.copyOfRange(outputsIris, 50, 70);

		double[][] inputsSpiral = ANN.readPattern("src/ann/spiralinput.txt", "\\s\\s");
		double[][] outputsSpiral = ANN.readPattern("src/ann/spiraloutput.txt", "\\s\\s");

		ANN irisann = new ANN(1, 0.5, 0.5, 4, 3, 3);
		ANN spiralann = new ANN(1, 0.5, 0.5, 2, 15, 1);
		ANN randann = new ANN(1, 0.05, 0.9, 4, 3, 3); /* new ANN(1, 0.05, 0.9, 1, 20, 1); /new ANN(1, 0.3, 0.5, 32, 16, 32);*/ //robins // IS IT THOUGH?!

		int basePopSize = 20; // 20 Robins
		int serialPopSize = 10; // 10 Robins
		double maxError = 0.3;
		int bufferSize = 3; // 3 Robins
		int pseudopopSize = 8;// 8, 32, 128 Robins

		double[][] randin = new double[20][4];
		double[][] randout = new double[20][3];
		int[] hlsizes = {1, 2, 5, 10, 25, 100};

		//findBestOverfittingByHL("thesis/spiraloverfitHL.txt", 1, 200000, 2000, spiralann, hlsizes, inputsSpiral, outputsSpiral);
		//findBestHL("thesis/spiralHL.txt", 20, 70000, 2000, spiralann, hlsizes, inputsSpiral, outputsSpiral);
		//trackError("thesis/spiraltrack.txt", 20, 100000, 2000, spiralann, inputsSpiral, outputsSpiral);
		//noRehearsalSerialLearning("thesis/norehearsal.txt", irisann, randin, randout, basePopSize, serialPopSize, maxError);
		//fullRehearsalSerialLearning("thesis/fullrehearsal.txt", irisann, inputsIris, outputsIris, basePopSize, serialPopSize, maxError);
		//randomRehearsalSerialLearning("thesis/dRRiris.txt", irisann, inputsIris, outputsIris, 
		//		basePopSize, serialPopSize, maxError, bufferSize);
		//sweepPseudoRehearsalSerialLearning("thesis/dSPRiris.txt", randann, 32, 32,
		//basePopSize, serialPopSize, maxError, bufferSize);
		//noThenFullThenRandomThenSweepPs("thesis/nothenfullthenrandomthensweepps.txt", randann, basePopSize, serialPopSize, maxError);
		
		basePopSize = 30;
		serialPopSize = 20;
		maxError = 0.01;
		bufferSize = 5;
		pseudopopSize = 128;
		noThenFullThenSweepThenSweepPs("thesis/eoycompare.txt", randann, basePopSize, serialPopSize, maxError, pseudopopSize, bufferSize, true);


		double[][] inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
		double[][] outputs = {{0}, {1}, {1}, {0}};
		ANN dynann = new ANN(1, 0.5, 0.5, 2, 1, 1);
		double dynamicError = 0.01;
		double abortlim = 10000;
		boolean reset = true;
		boolean reals = false;

		double[][] basein = {{0,0}, {0,1}};
		double[][] baseout = {{0}, {1}};
		double[][] serialin = {{1,0}, {1,1}};
		double[][] serialout = {{1}, {0}};
		//dynamicTrain("thesis/dynamictrack.txt", dynann, inputs, outputs, dynamicError, false);
		// TRY CHANGING BUFFER SIZE
		//sweepPseudoRehearsalSerialLearningDynVsNot("thesis/dynVsNot.txt", dynann, 
		//	basein, baseout, serialin, serialout, dynamicError, bufferSize, abortlim, reset, reals);

		//sweepPseudoRehearsalSerialLearningDynVsNot("thesis/dynVsNotReals.txt", dynann, 
		//		basein, baseout, serialin, serialout, dynamicError, bufferSize, abortlim, true, true);

		/*ANN dynIrisAnn = new ANN(1, 0.5, 0.5, 4, 1, 3);
		double irisError = 0.1;

		double[][] inputsIrisSmall = ANN.readPattern("src/ann/smallirisin.txt", "\\s\\s");
		double[][] outputsIrisSmall = ANN.readPattern("src/ann/smallirisout.txt", "\\s\\s");

		double[][] baseinputsIris = Arrays.copyOf(inputsIrisSmall, 20);
		double[][] baseoutputsIris = Arrays.copyOf(outputsIrisSmall, 20);
		double[][] serialinputsIris = Arrays.copyOfRange(inputsIrisSmall, 21, 30);
		double[][] serialoutputsIris = Arrays.copyOfRange(outputsIrisSmall, 21, 30);

		System.out.println(Arrays.deepToString(baseinputsIris));
		System.out.println(Arrays.deepToString(baseoutputsIris));
		System.out.println(Arrays.deepToString(serialinputsIris));
		System.out.println(Arrays.deepToString(serialoutputsIris));

		sweepPseudoRehearsalSerialLearningDynVsNot("thesis/dynVsNot.txt", dynIrisAnn, 
				baseinputsIris, baseoutputsIris, serialinputsIris, serialoutputsIris, irisError, bufferSize, abortlim, reset, reals);*/

		//findBestInputVectorSizePseudo("thesis/vectorSizes.txt",20,0,0,dynann,null);
		//findBestInputRealRangePseudo("thesis/rangeSizes.txt", 20, 0, 0, dynann, 3);
	}

	public static void findBestInputRealRangePseudo(String filename, int reps, int epochsPerRep, int printFrequency, 
			ANN ann, int vectorSize) throws IOException {

		ANN psnn = new ANN(1, 0.5, 0.5, 3, 4, 1);

		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		double maxError = 0.02;
		int bufferSize = 4;
		boolean reals = true;

		double[][] inputsCon = {{0,0,0}, {0,0,1}, {0,1,0}};
		double[][] outputsCon = {{0}, {1}, {1}};
		double[][] inputsPs = {{0,1,1}};
		double[][] outputsPs = {{0}};

		double[] ranges = {0.0, 0.2, 0.4, 0.8, 1.0, 2.0};
		
		double[][] data = new double[(inputsCon.length+inputsPs.length)*reps][ranges.length+1];

		for (int range = 0; range < ranges.length; range++) {

			for (int rep = 0; rep < reps; rep++) {
				psnn.reset();
				double min = 0.5 - (ranges[range]/2);
				double max = 0.5 + (ranges[range]/2);
				System.err.println("range " + ranges[range] + " min " + min + " max " + max);
				double[] trial = psnn.sweepPseudoRehearsalSerialLearning(inputsCon, outputsCon, 
						inputsPs, outputsPs, maxError, bufferSize, reals, min, max);

				Pattern p = new Pattern(inputsPs[0], outputsPs[0]);
				psnn.pass(p);
				for (int i = 0; i < inputsCon.length; i++) {
					Pattern tria = new Pattern(inputsCon[i], null);
					psnn.pass(tria);
					System.out.println(Arrays.toString(inputsCon[i]) + 
						" -> " + Arrays.toString(psnn.getCurrentOutput()));
					double[] out = psnn.getCurrentOutput();
					//f.write(i + "");
					//for (int j = 0; j <= range; j++) f.write("\t");
					//f.write(out[0] + "\n");
					int index = rep*(inputsCon.length+inputsPs.length) + i;
					data[index][0] = i;
					data[index][range+1] = out[0];
				}
				for (int i = 0; i < inputsPs.length; i++) {
					Pattern tria = new Pattern(inputsPs[i], null);
					psnn.pass(tria);
					System.out.println(Arrays.toString(inputsPs[i]) + 
						" -> " + Arrays.toString(psnn.getCurrentOutput()));
					double[] out = psnn.getCurrentOutput();
//					f.write(i+inputsCon.length + "");
//					for (int j = 0; j <= range; j++) f.write("\t");
//					f.write(out[0] + "\n");
					int index = rep*(inputsCon.length+inputsPs.length) + inputsCon.length + i;
					data[index][0] = inputsCon.length + i;
					data[index][range+1] = out[0];
				}
			}
		}
		for (double[] ad : data) {
			for (double d : ad) {
				f.write(d + "\t");
			}
			f.write("\n");
		}
		f.close();
	}

	public static void findBestInputVectorSizePseudo(String filename, int reps, int epochsPerRep, int printFrequency,
			ANN ann, int[] vectorSizes) throws IOException {
		// metric is by retention after 1000 reps

		ANN psnn = new ANN(1, 0.5, 0.5, 2, 4, 1);

		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());
		
		int numtests = 6;
				
		double[][] inputsConTwo = {{0,0}, {0,1}, {1,0}};
		double[][] outputsConTwo = {{0}, {1}, {1}};
		double[][] inputsPsTwo = {{1,1}};
		double[][] outputsPsTwo = {{0}};

		double[][] inputsConThree = {{0,0,0}, {0,0,1}, {0,1,0}};
		double[][] outputsConThree = {{0}, {1}, {1}};
		double[][] inputsPsThree = {{0,1,1}};
		double[][] outputsPsThree = {{0}};

		double[][] inputsConFour = {{0,0,0,0}, {0,0,0,1}, {0,0,1,0}};
		double[][] outputsConFour = {{0}, {1}, {1}};
		double[][] inputsPsFour = {{0,0,1,1}};
		double[][] outputsPsFour = {{0}};

		double[][] inputsConFive = {{0,0,0,0,0}, {0,0,0,0,1}, {0,0,0,1,0}};
		double[][] outputsConFive = {{0}, {1}, {1}};
		double[][] inputsPsFive = {{0,0,0,1,1}};
		double[][] outputsPsFive = {{0}};
		
		double[][] data = new double[(inputsConTwo.length+inputsPsTwo.length)*reps][numtests+1];

		double maxError = 0.02;
		int bufferSize = 4;
		boolean reals = false;
		//double[] trial = psnn.sweepPseudoRehearsalSerialLearning(inputsConTwo, outputsConTwo, 
		//	inputsPsTwo, outputsPsTwo, maxError, bufferSize, reals);

		/*for (int i = 0; i < trial.length; i++) {
			f.write(i + "\t" + trial[i] + "\n");
		}

		f.close();*/
		//f.write("1");
		int vecSize = 1;

		for (int rep = 0; rep < reps; rep++) {
			psnn.reset();
			double[] trial = psnn.sweepPseudoRehearsalSerialLearning(inputsConTwo, outputsConTwo, 
					inputsPsTwo, outputsPsTwo, maxError, bufferSize, reals, 0 , 1);

			Pattern p = new Pattern(inputsPsTwo[0], outputsPsTwo[0]);
			psnn.pass(p);
			for (int i = 0; i < inputsConTwo.length; i++) {
				Pattern tria = new Pattern(inputsConTwo[i], null);
				psnn.pass(tria);
				//System.out.println(Arrays.toString(inputsConTwo[i]) +
				//	" -> " + Arrays.toString(psnn.getCurrentOutput()));
				double[] out = psnn.getCurrentOutput();
				//f.write("\t" + out[0]);
				int index = rep*(inputsConTwo.length+inputsPsTwo.length) + i;
				data[index][0] = i;
				data[index][vecSize] = out[0];
			}
			for (int i = 0; i < inputsPsTwo.length; i++) {
				Pattern tria = new Pattern(inputsPsTwo[i], null);
				psnn.pass(tria);
				//System.out.println(Arrays.toString(inputsPsTwo[i]) + 
				//	" -> " + Arrays.toString(psnn.getCurrentOutput()));
				double[] out = psnn.getCurrentOutput();
				//f.write("\t" + out[0]);
				int index = rep*(inputsConTwo.length+inputsPsTwo.length) + inputsConTwo.length + i;
				data[index][0] = inputsConTwo.length + i;
				data[index][vecSize] = out[0];
			}
		}
		vecSize++;

		//f.write("\n3");
		psnn = new ANN(1, psnn.learningSpeed, psnn.momentum, 3, 4, 1);

		for (int rep = 0; rep < reps; rep++) {
			psnn.reset();
			double[] trial = psnn.sweepPseudoRehearsalSerialLearning(inputsConThree, outputsConThree, 
					inputsPsThree, outputsPsThree, maxError, bufferSize, reals, 0 , 1);

			Pattern p = new Pattern(inputsPsThree[0], outputsPsThree[0]);
			psnn.pass(p);
			for (int i = 0; i < inputsConThree.length; i++) {
				Pattern tria = new Pattern(inputsConThree[i], null);
				psnn.pass(tria);
				//System.out.println(Arrays.toString(inputsConTwo[i]) +
				//	" -> " + Arrays.toString(psnn.getCurrentOutput()));
				double[] out = psnn.getCurrentOutput();
				//f.write("\t" + out[0]);
				int index = rep*(inputsConTwo.length+inputsPsTwo.length) + i;
				data[index][0] = i;
				data[index][vecSize] = out[0];

			}
			for (int i = 0; i < inputsPsThree.length; i++) {
				Pattern tria = new Pattern(inputsPsThree[i], null);
				psnn.pass(tria);
				//System.out.println(Arrays.toString(inputsPsTwo[i]) + 
				//	" -> " + Arrays.toString(psnn.getCurrentOutput()));
				double[] out = psnn.getCurrentOutput();
				//f.write("\t" + out[0]);
				int index = rep*(inputsConTwo.length+inputsPsTwo.length) + inputsConTwo.length + i;
				data[index][0] = inputsConTwo.length + i;
				data[index][vecSize] = out[0];
			}
		}
		vecSize++;

		//f.write("\n4");
		psnn = new ANN(1, psnn.learningSpeed, psnn.momentum, 4, 4, 1);

		for (int rep = 0; rep < reps; rep++) {
			psnn.reset();
			double[] trial = psnn.sweepPseudoRehearsalSerialLearning(inputsConFour, outputsConFour, 
					inputsPsFour, outputsPsFour, maxError, bufferSize, reals, 0 , 1);

			Pattern p = new Pattern(inputsPsFour[0], outputsPsFour[0]);
			psnn.pass(p);
			for (int i = 0; i < inputsConThree.length; i++) {
				Pattern tria = new Pattern(inputsConFour[i], null);
				psnn.pass(tria);
				//System.out.println(Arrays.toString(inputsConTwo[i]) +
				//	" -> " + Arrays.toString(psnn.getCurrentOutput()));
				double[] out = psnn.getCurrentOutput();
				int index = rep*(inputsConTwo.length+inputsPsTwo.length) + i;
				data[index][0] = 7.5;
				data[index][vecSize] = out[0];
			}
			for (int i = 0; i < inputsPsFour.length; i++) {
				Pattern tria = new Pattern(inputsPsFour[i], null);
				psnn.pass(tria);
				//System.out.println(Arrays.toString(inputsPsTwo[i]) + 
				//	" -> " + Arrays.toString(psnn.getCurrentOutput()));
				double[] out = psnn.getCurrentOutput();
				//f.write("\t" + out[0]);
				int index = rep*(inputsConTwo.length+inputsPsTwo.length) + inputsConTwo.length + i;
				data[index][0] = inputsConTwo.length + i;
				data[index][vecSize] = out[0];
			}
		}
		vecSize++;
		
		//f.write("\n5");

		psnn = new ANN(1, psnn.learningSpeed, psnn.momentum, 5, 4, 1);
		for (int rep = 0; rep < reps; rep++) {
			psnn.reset();
			double[] trial = psnn.sweepPseudoRehearsalSerialLearning(inputsConFive, outputsConFive, 
					inputsPsFive, outputsPsFive, maxError, bufferSize, reals, 0 , 1);

			Pattern p = new Pattern(inputsPsFive[0], outputsPsFive[0]);
			psnn.pass(p);
			for (int i = 0; i < inputsConFive.length; i++) {
				Pattern tria = new Pattern(inputsConFive[i], null);
				psnn.pass(tria);
				//System.out.println(Arrays.toString(inputsConTwo[i]) +
				//	" -> " + Arrays.toString(psnn.getCurrentOutput()));
				double[] out = psnn.getCurrentOutput();
				int index = rep*(inputsConTwo.length+inputsPsTwo.length) + i;
				data[index][0] = 7.5;
				data[index][vecSize] = out[0];
			}
			for (int i = 0; i < inputsPsFive.length; i++) {
				Pattern tria = new Pattern(inputsPsFive[i], null);
				psnn.pass(tria);
				//System.out.println(Arrays.toString(inputsPsTwo[i]) + 
				//	" -> " + Arrays.toString(psnn.getCurrentOutput()));
				double[] out = psnn.getCurrentOutput();
				//f.write("\t" + out[0]);
				int index = rep*(inputsConTwo.length+inputsPsTwo.length) + inputsConTwo.length + i;
				data[index][0] = inputsConTwo.length + i;
				data[index][vecSize] = out[0];
			}
		}
		vecSize++;

		reals = true;

		//f.write("\n1");

		psnn = new ANN(1, psnn.learningSpeed, psnn.momentum, 2, 4, 1);

		for (int rep = 0; rep < reps; rep++) {
			psnn.reset();
			double[] trial = psnn.sweepPseudoRehearsalSerialLearning(inputsConTwo, outputsConTwo, 
					inputsPsTwo, outputsPsTwo, maxError, bufferSize, reals, 0 , 1);

			Pattern p = new Pattern(inputsPsTwo[0], outputsPsTwo[0]);
			psnn.pass(p);
			for (int i = 0; i < inputsConTwo.length; i++) {
				Pattern tria = new Pattern(inputsConTwo[i], null);
				psnn.pass(tria);
				//System.out.println(Arrays.toString(inputsConTwo[i]) +
				//	" -> " + Arrays.toString(psnn.getCurrentOutput()));
				double[] out = psnn.getCurrentOutput();
				int index = rep*(inputsConTwo.length+inputsPsTwo.length) + i;
				data[index][0] = 7.5;
				data[index][vecSize] = out[0];
			}
			for (int i = 0; i < inputsPsTwo.length; i++) {
				Pattern tria = new Pattern(inputsPsTwo[i], null);
				psnn.pass(tria);
				//System.out.println(Arrays.toString(inputsPsTwo[i]) + 
				//	" -> " + Arrays.toString(psnn.getCurrentOutput()));
				double[] out = psnn.getCurrentOutput();
				int index = rep*(inputsConTwo.length+inputsPsTwo.length) + inputsConTwo.length + i;
				data[index][0] = index;
				data[index][vecSize] = out[0];
				//f.write("\t" + out[0]);
			}
		}
		vecSize++;

		//f.write("\n5");

		psnn = new ANN(1, psnn.learningSpeed, psnn.momentum, 5, 4, 1);
		for (int rep = 0; rep < reps; rep++) {
			psnn.reset();
			double[] trial = psnn.sweepPseudoRehearsalSerialLearning(inputsConFive, outputsConFive, 
					inputsPsFive, outputsPsFive, maxError, bufferSize, reals, 0 , 1);

			Pattern p = new Pattern(inputsPsFive[0], outputsPsFive[0]);
			psnn.pass(p);
			for (int i = 0; i < inputsConFive.length; i++) {
				Pattern tria = new Pattern(inputsConFive[i], null);
				psnn.pass(tria);
				//System.out.println(Arrays.toString(inputsConTwo[i]) +
				//	" -> " + Arrays.toString(psnn.getCurrentOutput()));
				double[] out = psnn.getCurrentOutput();
				int index = rep*(inputsConTwo.length+inputsPsTwo.length) + i;
				data[index][0] = i;
				data[index][vecSize] = out[0];
			}
			for (int i = 0; i < inputsPsFive.length; i++) {
				Pattern tria = new Pattern(inputsPsFive[i], null);
				psnn.pass(tria);
				//System.out.println(Arrays.toString(inputsPsTwo[i]) + 
				//	" -> " + Arrays.toString(psnn.getCurrentOutput()));
				double[] out = psnn.getCurrentOutput();
				//f.write("\t" + out[0]);
				int index = rep*(inputsConTwo.length+inputsPsTwo.length) + inputsConTwo.length + i;
				data[index][0] = inputsConTwo.length + i;
				data[index][vecSize] = out[0];
			}
		}
		
		for (double[] ad : data) {
			for (double d : ad) {
				f.write(d + "\t");
			}
			f.write("\n");
		}


		f.close();
		// for each vector size
		// learn three of the things concurrently
		// try to pseudorehearse a fourth for 1000 reps
		// output learnedness of the new thing
	}

	public static void findBestOverfittingByHL(String filename, int reps, int epochsPerRep, int printFrequency,
			ANN ann, int[] hlsizes, double[][] inputs, double[][] outputs) throws IOException {
		int numDataPoints = epochsPerRep/printFrequency;
		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		double[][] trainin = new double[inputs.length/3][inputs[0].length];
		double[][] trainout = new double[outputs.length/3][outputs[0].length];
		double[][] testin = new double[inputs.length-trainin.length][inputs[0].length];
		double[][] testout = new double[outputs.length-trainout.length][outputs[0].length];

		ANN.randomTrainPopulation(inputs, trainin, testin, outputs, trainout, testout);

		System.err.println(Arrays.deepToString(trainin) + "\n"
				+ Arrays.deepToString(trainout));

		f.write("0\t");
		for (int i = 0; i < epochsPerRep; i++) {
			if (i%printFrequency == 0) f.write(i + "\t");
		}
		f.write("\n");
		for (int hlsize = 0; hlsize < hlsizes.length; hlsize++) {
			ann = new ANN(1, ann.learningSpeed, ann.momentum, ann.layerSize(0), hlsizes[hlsize], ann.layerSize(2));
			System.err.println(ann.layerSize(1));
			double[][] tracks = new double[numDataPoints][reps];
			for (int q = 0; q < reps; q++) {
				ann.reset();

				ANN.randomTrainPopulation(inputs, trainin, testin, outputs, trainout, testout);

				int dataPoint = 0;
				for (int i = 0; i < epochsPerRep; i++) {
					ANN.permute(trainin, trainout);
					for (int j = 0; j < trainin.length; j++) {
						Pattern trial = new Pattern(trainin[j], trainout[j]);
						ann.pass(trial);
						ann.updateAllWeights(trial);
					}
					double err = ANN.populationError(ann.outputs(testin), testout);
					if (i%printFrequency == 0) {
						tracks[dataPoint][q] = err;
						dataPoint++;
					}
				}
			}
			f.write(hlsizes[hlsize] + "\t");
			for (int dp = 0; dp < tracks.length; dp++) {
				f.write(SimsA01.median(tracks[dp]) + "\t");
				//System.err.println(median(tracks[dp]));
			}
			f.write("\n");
		}
		f.close();
	}

	public static void findBestHL(String filename, int reps, int epochsPerRep, int printFrequency,
			ANN ann, int[] hlsizes, double[][] inputs, double[][] outputs) throws IOException {
		int numDataPoints = epochsPerRep/printFrequency;
		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());
		/*
		double[][] trainin = new double[inputs.length/3][inputs[0].length];
		double[][] trainout = new double[outputs.length/3][outputs[0].length];
		double[][] testin = new double[inputs.length-trainin.length][inputs[0].length];
		double[][] testout = new double[outputs.length-trainout.length][outputs[0].length];

		ANN.randomTrainPopulation(inputs, trainin, testin, outputs, trainout, testout);

		System.err.println(Arrays.deepToString(trainin) + "\n"
				+ Arrays.deepToString(trainout));*/

		f.write("0\t");
		for (int i = 0; i < epochsPerRep; i++) {
			if (i%printFrequency == 0) f.write(i + "\t");
		}
		f.write("\n");
		for (int hlsize = 0; hlsize < hlsizes.length; hlsize++) {
			ann = new ANN(1, ann.learningSpeed, ann.momentum, ann.layerSize(0), hlsizes[hlsize], ann.layerSize(2));
			System.err.println(ann.layerSize(1));
			double[][] tracks = new double[numDataPoints][reps];
			for (int q = 0; q < reps; q++) {
				ann.reset();

				int dataPoint = 0;
				for (int i = 0; i < epochsPerRep; i++) {
					ANN.permute(inputs, outputs);
					for (int j = 0; j < inputs.length; j++) {
						Pattern trial = new Pattern(inputs[j], outputs[j]);
						ann.pass(trial);
						ann.updateAllWeights(trial);
					}
					double err = ANN.populationError(ann.outputs(inputs), outputs);
					if (i%printFrequency == 0) {
						tracks[dataPoint][q] = err;
						dataPoint++;
					}
				}
			}
			f.write(hlsizes[hlsize] + "\t");
			for (int dp = 0; dp < tracks.length; dp++) {
				f.write(SimsA01.median(tracks[dp]) + "\t");
				//System.err.println(dp + "/" + tracks.length + ": " + SimsA01.median(tracks[dp]));
			}
			f.write("\n");
		}
		f.close();
	}

	public static void trackError(String filename, int reps, int epochsPerRep, int printFrequency,
			ANN ann, double[][] inputs, double[][] outputs) throws IOException {
		File file = new File(filename); //new File("omg_ann_or.txt");
		FileWriter f = new FileWriter(file.getAbsolutePath());
		f.write("0\t");
		for (int i = 0; i < epochsPerRep; i++) {
			if (i%printFrequency == 0) f.write(i + "\t");
		}
		f.write("\n");
		for (int q = 0; q < reps; q++) {
			f.write(q + "\t");
			ann.reset();
			for (int i = 0; i < epochsPerRep; i++) {
				double[][] currentoutputs = new double[inputs.length][outputs[0].length];
				ANN.permute(inputs, outputs);
				for (int j = 0; j < inputs.length; j++) {
					Pattern trial = new Pattern(inputs[j], outputs[j]);
					ann.pass(trial);
					ann.updateAllWeights(trial);
					double[] currentoutput = ann.getCurrentOutput();
					currentoutputs[j] = currentoutput;
				}
				double err = ANN.populationError(ann.outputs(inputs), outputs);
				if (i%printFrequency == 0) {
					/*f.write(Arrays.deepToString(inputs)+ " " + Arrays.deepToString(ann.outputs(inputs))+"\t");//*/
					f.write(err + "\t");
					System.err.println(i + ": " + err);
				}
			}
			f.write("\n");
		}
		f.close();
	}

	public static void createRandomDataSet(double[][] inputs, double[][] outputs) {
		Random r = new Random();

		for (int i = 0; i < inputs.length; i++) {
			for (int j = 0; j < inputs[0].length; j++) {
				inputs[i][j] = r.nextDouble() < 0.5 ? 0 : 1;
			}
		}

		for (int i = 0; i < outputs.length; i++) {
			for (int j = 0; j < outputs[0].length; j++) {
				outputs[i][j] = r.nextDouble() < 0.5 ? 0 : 1;
			}
		}
	}

	public static void sweepPseudoRehearsalSerialLearning(String filename, ANN ann, int inputLength, int outputLength,
			int basePopSize, int serialPopSize, double maxError, int bufferSize, boolean reals) throws IOException {
		double[][] inputs = new double[basePopSize+serialPopSize][inputLength];
		double[][] outputs = new double[basePopSize+serialPopSize][outputLength];
		createRandomDataSet(inputs, outputs);

		double[][] basein = new double[basePopSize][inputs[0].length];
		double[][] baseout = new double[basePopSize][outputs[0].length];
		double[][] serialin = new double[serialPopSize][inputs[0].length];
		double[][] serialout = new double[serialPopSize][outputs[0].length];

		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		int reps = 50; // number of tests
		double[][] allout = new double[serialin.length+1][reps]; // holds all test data

		for (int rep = 0; rep < reps; rep++) {
			System.out.println("one: trial " + rep);
			ann.reset();
			createRandomDataSet(inputs, outputs);

			basein = new double[basePopSize][inputs[0].length];
			baseout = new double[basePopSize][outputs[0].length];
			serialin = new double[serialPopSize][inputs[0].length];
			serialout = new double[serialPopSize][outputs[0].length];

			ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout); // %%%%REALS

			double[] trial = ann.sweepPseudoRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError, bufferSize, reals, 0 , 1);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
			System.out.println("two: finished trial " + rep);
		}
		System.out.println("three: writing to file");

		for (int i = 0; i < allout.length; i++) {
			f.write(i + "\t" + SimsA01.median(allout[i]) + "\n");
		}

		f.close();
		System.out.println("done");

	}

	public static void sweepPseudoRehearsalSerialLearningDynVsNot(String filename, ANN ann, 
			double[][] basein, double[][] baseout,
			double[][] serialin, double[][] serialout, 
			double maxError, int bufferSize, double abortlim,
			boolean reset, boolean reals) throws IOException {

		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		/*f.write("0\t");
		for (int i = 0; i < serialout.length + 1; i++) {
			f.write(i + "\t");
		}
		f.write("\n");*/

		int reps = 1; // number of tests
		double[][] allout = new double[serialin.length+1][reps]; // holds all test data

		for (int rep = 0; rep < reps; rep++) {
			//System.out.println("one: trial " + rep);
			ann.reset();

			double[] trial = ann.sweepPseudoRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError, bufferSize, reals, 0 , 1);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
			//System.out.println("two: finished trial " + rep);
		}
		//System.out.println("three: writing to file");

		/*f.write("normal\t");
		for (int i = 0; i < allout.length; i++) {
			f.write(SimsA01.median(allout[i]) + "\t");
		}
		f.write("\n");*/


		//// dynamic! MUST GO AFTER BECAUSE RESET DOESN'T RESENT # OF HL NODES!!!!!!!
		System.out.println("\n\nDYNAMICCCCC");
		double[][][] dynamOut = new double[2][serialin.length+1][reps]; 

		for (int rep = 0; rep < reps; rep++) {
			//System.out.println("one: trial " + rep);
			ann.reset();

			double[][] trial = ann.dynamicSweepPseudo(basein, baseout, serialin, serialout, maxError, bufferSize, abortlim, reset, reals);

			System.out.println("trials:");
			System.out.println(Arrays.deepToString(trial) + "\n");
			for (int j = 0; j < dynamOut.length; j++) {
				dynamOut[0][j][rep] = trial[0][j];
			}
			for (int j = 0; j < dynamOut.length; j++) {
				dynamOut[1][j][rep] = trial[1][j];
			}
			//System.out.println("two: finished trial " + rep);
		}
		//System.out.println("three: writing to file");

		f.write("dynamic base\t");
		for (int i = 0; i < dynamOut[0].length; i++) {
			f.write(SimsA01.median(dynamOut[0][i]) + "\t");
		}
		f.write("\n");

		f.write("dynamic serial\t");
		for (int i = 0; i < dynamOut[0].length; i++) {
			f.write(SimsA01.median(dynamOut[1][i]) + "\t");
		}
		f.write("\n");

		f.close();
		System.out.println("done");

	}

	public static void randomRehearsalSerialLearning(String filename, ANN ann,  double[][] inputs, double[][] outputs,
			int basePopSize, int serialPopSize, double maxError, int bufferSize) throws IOException {

		double[][] basein = new double[basePopSize][inputs[0].length];
		double[][] baseout = new double[basePopSize][outputs[0].length];
		double[][] serialin = new double[serialPopSize][inputs[0].length];
		double[][] serialout = new double[serialPopSize][outputs[0].length];


		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		int reps = 50;
		double[][] allout = new double[serialin.length+1][reps];

		for (int rep = 0; rep < reps; rep++) {
			ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);

			double[] trial = ann.randomRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError, bufferSize);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
		}

		for (int i = 0; i < allout.length; i++) {
			f.write(i + "\t" + SimsA01.median(allout[i]) + "\n");
		}

		f.close();

	}

	public static void noThenFullThenSweepThenSweepPs(String filename, ANN ann, 
			int basePopSize, int serialPopSize, double maxError, int popSize, int bufferSize, boolean reals) throws IOException {

		int inputSize = ann.layerSize(0);
		int outputSize = ann.layerSize(ann.numLayers()-1);

		double[][] inputs = new double[basePopSize+serialPopSize][inputSize];
		double[][] outputs = new double[basePopSize+serialPopSize][outputSize];

		double[][] basein = shortinputsIris;//new double[basePopSize][inputSize];
		double[][] baseout = shortoutputsIris;//new double[basePopSize][outputSize];
		double[][] serialin = inpseudoIris;//new double[serialPopSize][inputSize];
		double[][] serialout = outpseudoIris;//new double[serialPopSize][outputSize];

		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		f.write("0\t");
		for (int i = 0; i < serialout.length + 1; i++) {
			f.write(i + "\t");
		}
		f.write("\n");

		int reps = 50;

		double[][] allout = new double[serialin.length+1][reps];

		// no rehearsal /////////////////////
		System.out.println("no rehearsal...");

		for (int rep = 0; rep < reps; rep++) {
			System.out.println("rep " + rep);
			//createRandomData(inputs, outputs);
			ann.reset();
			//ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);

			double[] trial = ann.noRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
		}

		f.write("no rehearsal\t");
		for (int i = 0; i < allout.length; i++) {
			f.write(SimsA01.median(allout[i]) + "\t");
		}
		f.write("\n");

		// full rehearsal /////////////////
		System.out.println("full rehearsal...");

		allout = new double[serialin.length+1][reps];
		for (int rep = 0; rep < reps; rep++) {
			System.out.println("rep " + rep);
			//createRandomData(inputs, outputs);
			ann.reset();
			//ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);

			double[] trial = ann.fullRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
		}

		f.write("full rehearsal\t");
		for (int i = 0; i < allout.length; i++) {
			f.write(SimsA01.median(allout[i]) + "\t");
		}
		f.write("\n");

		/*
		// sweep //////////////////////
		System.out.println("aaaaaaand random....");
		allout = new double[serialin.length+1][reps];

		for (int rep = 0; rep < reps; rep++) {
			System.out.println("rep " + rep);
			createRandomData(inputs, outputs);
			ann.reset();
			ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);

			double[] trial = ann.sweepRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
		}

		f.write("random rehearsal\t");
		for (int i = 0; i < allout.length; i++) {
			f.write(SimsA01.median(allout[i]) + "\t");
		}
		f.write("\n");

		 */// sweep pseudo
		System.out.println("aaaaaaaaaaaaaaand sweep pseudorehearsal...!");

		allout = new double[serialin.length+1][reps]; 

		for (int rep = 0; rep < reps; rep++) {
			System.out.println("rep " + rep);
			ann.reset();
			/*createRandomDataSet(inputs, outputs);

			basein = new double[basePopSize][inputs[0].length];
			baseout = new double[basePopSize][outputs[0].length];
			serialin = new double[serialPopSize][inputs[0].length];
			serialout = new double[serialPopSize][outputs[0].length];

			ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout); // %%%%REALS*/

			double[] trial = ann.sweepPseudoRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError, popSize, bufferSize, reals, 0 , 1);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
		}

		f.write("sweep pseudorehearsal\t");
		for (int i = 0; i < allout.length; i++) {
			f.write(SimsA01.median(allout[i]) + "\t");
		}
		f.write("\n");

		f.close();

	}

	public static void fullRehearsalSerialLearning(String filename, ANN ann,  double[][] inputs, double[][] outputs,
			int basePopSize, int serialPopSize, double maxError) throws IOException {

		double[][] basein = new double[basePopSize][inputs[0].length];
		double[][] baseout = new double[basePopSize][outputs[0].length];
		double[][] serialin = new double[serialPopSize][inputs[0].length];
		double[][] serialout = new double[serialPopSize][outputs[0].length];

		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		int reps = 50;
		double[][] allout = new double[serialin.length+1][reps];

		for (int rep = 0; rep < reps; rep++) {
			ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);

			double[] trial = ann.fullRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
		}

		for (int i = 0; i < allout.length; i++) {
			f.write(i + "\t" + SimsA01.median(allout[i]) + "\n");
		}

		f.close();

	}

	public static void sweepRehearsalSerialLearning(String filename, ANN ann,  double[][] inputs, double[][] outputs,
			int basePopSize, int serialPopSize, double maxError) throws IOException {

		double[][] basein = new double[basePopSize][inputs[0].length];
		double[][] baseout = new double[basePopSize][outputs[0].length];
		double[][] serialin = new double[serialPopSize][inputs[0].length];
		double[][] serialout = new double[serialPopSize][outputs[0].length];

		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		int reps = 50;
		double[][] allout = new double[serialin.length+1][reps];

		for (int rep = 0; rep < reps; rep++) {
			ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);

			double[] trial = ann.sweepRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
		}

		for (int i = 0; i < allout.length; i++) {
			f.write(i + "\t" + SimsA01.median(allout[i]) + "\n");
		}

		f.close();

	}

	public static void noRehearsalSerialLearning(String filename, ANN ann,  double[][] inputs, double[][] outputs,
			int basePopSize, int serialPopSize, double maxError) throws IOException {

		double[][] basein = new double[basePopSize][inputs[0].length];
		double[][] baseout = new double[basePopSize][outputs[0].length];
		double[][] serialin = new double[serialPopSize][inputs[0].length];
		double[][] serialout = new double[serialPopSize][outputs[0].length];

		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		int reps = 5;
		double[][] allout = new double[serialin.length+1][reps];

		//createRandomData(inputs, outputs);
		//ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);

		for (int rep = 0; rep < reps; rep++) {

			//createRandomData(inputs, outputs);

			ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);
			ann.reset();

			double[] trial = ann.noRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
		}

		for (int i = 0; i < allout.length; i++) {
			f.write(i + "\t" + SimsA01.median(allout[i]) + "\n");
		}

		f.close();

	}

	public static void createRandomData(double[][] inputs, double[][] outputs) {
		Random r = new Random(); 
		for (int i = 0; i < inputs.length; i++) {
			for (int j = 0; j < inputs[i].length; j++) {
				inputs[i][j] = r.nextDouble() < 0.5 ? 0.0 : 1.0;
			}
			for (int j = 0; j < outputs[i].length; j++) {
				outputs[i][j] = r.nextDouble() < 0.5 ? 0.0 : 1.0;
			}
		}
	}

	public static void dynamicTrain(String filename, ANN ann, double[][] inputs, double[][] outputs, double maxerror, boolean reset) throws IOException {
		double[] errs = ann.dynamicTrain(inputs, outputs, maxerror, reset);

		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		for (int i = 0; i < errs.length; i++) {
			f.write((i*100) + "\t");
		}
		f.write("\n");
		for (int i = 0; i < errs.length; i++) {
			f.write(errs[i] + "\t");
		}
		f.write("\n");

		f.close();

	}

	public static void dynamicSweepPseudo(ANN ann, double[][] baseinputs, double[][] baseoutputs, 
			double[][] serialinputs, double[][] serialoutputs, double maxerror) {

	}
}
