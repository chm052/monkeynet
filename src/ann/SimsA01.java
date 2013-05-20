package ann;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

public class SimsA01 {

	public static void main(String[] args) throws IOException {
		double[][] inputsFull = {{0, 0}, {0, 1.0}, {1.0, 0}, {1.0, 1.0}};
		double[][] outputsXOR = {{0.0}, {1.0}, {1.0}, {0.0}};
		double[][] outputsOR = {{0.0}, {1.0}, {1.0}, {1.0}};

		double[][] inputsIris = ANN.readPattern("src/ann/irisinputs.txt", "\\s\\s");
		double[][] outputsIris = ANN.readPattern("src/ann/irisoutputs.txt", "\\s\\s");
		double[][] shortinputsIris = Arrays.copyOf(inputsIris, 3);
		double[][] shortoutputsIris = Arrays.copyOf(outputsIris, 3);

		ANN ann = new ANN(1, 0.1, 0.5, 2, 2, 1);
		ANN irisann = new ANN(1, 0.5, 0.5, 4, 3, 3);
		ANN testan = new ANN(1, 0.8, 0.9, 1, 5, 1);

		double[][] ins = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
		double[][] xtox = {{0.0}, {1.0}, {1.0}, {1.0}};

		//trackError("bob.txt", 10, 40000, 5000, testan, ins, xtox);

		/*
		double err = 0;
		for (int i = 0; i < 1000; i++) {
			testan.reset();
			for (int j = 0; j < ins.length; j++) {
				Pattern trial = new Pattern(ins[j], xtox[j]);
				testan.pass(trial);
				testan.updateAllWeights(trial);
				System.out.println("\t"+Arrays.toString(ann.getCurrentOutput()));//(err);

			}
			err = populationError(ann.outputs(ins), xtox);
			System.out.println(Arrays.toString(ann.getCurrentOutput()));//(err);
		}*/

		double[] noises = {0.0, 0.2, 0.4};
		double[] mentums = {0.1, 0.9};
		int[] hlsizes = /*{2,4};//*/{2, 3, 4, 5, 6};
		irisann = new ANN(1, 0.5, 0.5, 4, 3, 3);
		//findBestOverfittingByHL("overfit_hl.txt", 1, 200, 10, irisann, hlsizes, inputsIris, outputsIris);
		//trackCrossValidation("cross_iris.txt", 5, 2000, 100, irisann, inputsIris, outputsIris);
		//compareMomentumHL("iris_mom_hl.txt", 1000, 100, 2, irisann, mentums, hlsizes, inputsIris, outputsIris);
		// findBestNoiseForGen("noise_avg_iris.txt", 200, 0.5, irisann, inputsIris, outputsIris);
		// trackNoiseGeneralisation("noise_iris.txt", 20, 500, 5, 0.1, irisann, noises, inputsIris, outputsIris);
		// trackError("iris_track_mom_05.txt", 20, 70, 1, 0.5, irisann, inputsIris, outputsIris);
		// findBestMomentum("iris_mentums.txt", 40, 70, 1, irisann, inputsIris, outputsIris);
		// localMinPointsXOR("local_mins.txt", 100, 5000, 0.05, ann, inputsFull, outputsXOR);
		//trackError("GERR_xor_track_mom_01.txt", 50, 5000, 500, ann, inputsFull, outputsXOR);

		createRandomDataSet("src/ann/custominputs.txt", "src/ann/customoutputs.txt", 500);
		double[][] customIn = ANN.readPattern("src/ann/custominputs.txt", "\\s\\s");
		double[][] customOut = ANN.readPattern("src/ann/customoutputs.txt", "\\s\\s");
		ANN customann = new ANN(1, 0.9, 0.9, 2, 2, 1);
		ANN noiseann = new ANN(1, 0.1, 0.5, 2, 3, 1);
		double[] lisps = {0.1, 0.9};
		double[] mentums2 = {0.1, 0.5, 0.9};
		//compareMomentumLearningSpeed("ML_medians_custom.txt", 500, 200, 10, customann, mentums2, lisps, customIn, customOut);
		// findBestOverfittingByHL("custom_overfit_hl.txt", 5000, 300, 5, customann, hlsizes, customIn, customOut);
		//findBestOverfittingByHL("custom_overfit_hl_refresh.txt", 5000, 300, 5, customann, hlsizes, customIn, customOut);
		findBestOverfittingByHL("AAAA.txt", 500, 300, 5, irisann, hlsizes, inputsIris, outputsIris);

		//scatterRandomOutputsByHL("out_scatter.txt", 300, 50, 40, customann, hlsizes);
		//trackError("iris_track_noise.txt", 20, 70, 1, irisann, inputsIris, outputsIris);
		// trackError("axor_track_noise.txt", 50, 5000, 500, noiseann, inputsFull, outputsXOR);
		//compareHiddenLayerSizes("AAHLsize_medians_iris.txt", 500, 70, 1, irisann, hlsizes, inputsIris, outputsIris);

		/*
		compareMomentumLearningSpeed("ML_medians_or.txt", 50, 2000, 100, inputsFull, outputsOR);
		compareMomentumLearningSpeed("ML_medians.txt", 50, 5000, 500, inputsFull, outputsXOR);
		compareMomentumLearningSpeed("ML_medians_iris.txt", 50, 200, 5, irisann, inputsIris, outputsIris); */

		//compareHiddenLayerSizes("HLsize_medians.txt", 100, 2500, 100, 0.5, inputsFull, outputsXOR);
		//compareHiddenLayerSizes("HLsize_medians_iris.txt", 500, 70, 1, 0.5, irisann, inputsIris, outputsIris);

		/*findBestMomentum("DEC_xor_momentum_medians.txt", 100, 8500, 500, inputsFull, outputsXOR); */
	}
	
	public static void scatterRandomOutputsByHL(String filename, int epochs, int printFrequency, int pointsPrinted, ANN ann, int[] hlsizes) throws IOException {
		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());
		createRandomDataSet("src/ann/temprandomin.txt", "src/ann/temprandomout.txt", 500);
		Random r = new Random();
		double[][] inputs = ANN.readPattern("src/ann/temprandomin.txt", "\\s\\s");
		double[][] outputs = ANN.readPattern("src/ann/temprandomout.txt", "\\s\\s");
		
		double[][] trainin = new double[inputs.length/3][inputs[0].length];
		double[][] trainout = new double[outputs.length/3][outputs[0].length];
		double[][] testin = new double[inputs.length-trainin.length][inputs[0].length];
		double[][] testout = new double[outputs.length-trainout.length][outputs[0].length];
		
		ANN.randomTrainPopulation(inputs, trainin, testin, outputs, trainout, testout);
		int numDataPoints = epochs/printFrequency;
		f.write("0\t");
		for (int ndp = 0; ndp < numDataPoints; ndp++) {
			for (int pp = 0; pp < pointsPrinted; pp++) {
				f.write(ndp + "\t");
			}
		}
		f.write("\n");
		for (int hlsize = 0; hlsize < hlsizes.length; hlsize++) {
			f.write(hlsizes[hlsize] + "\t");
			ann = new ANN(1, ann.learningSpeed, ann.momentum, ann.layerSize(0), hlsizes[hlsize], ann.layerSize(2));
			System.err.println(ann.layerSize(1));
			for (int i = 0; i < epochs; i++) {
				ANN.permute(trainin, trainout);
				for (int j = 0; j < trainin.length; j++) {
					Pattern trial = new Pattern(trainin[j], trainout[j]);
					ann.pass(trial);
					ann.updateAllWeights(trial);
				}
				double[][] outs = ann.outputs(testin);
				if (i%printFrequency == 0) {
					for (int kk = 0; kk < pointsPrinted; kk++) {
						f.write(outs[r.nextInt(outs.length)][0] + "\t");
					}
				}
			}
			f.write("\n");
			
		}
		f.close();
		
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
				//createRandomDataSet("src/ann/custominputs.txt", "src/ann/customoutputs.txt", 500);
				//inputs = ANN.readPattern("src/ann/custominputs.txt", "\\s\\s");
				//outputs = ANN.readPattern("src/ann/customoutputs.txt", "\\s\\s");
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
				f.write(median(tracks[dp]) + "\t");
				//System.err.println(median(tracks[dp]));
			}
			f.write("\n");
		}
		f.close();
	}

	public static void createRandomDataSet(String in, String out, int size) throws IOException {
		File infile = new File(in);
		File outfile = new File(out);

		FileWriter inwriter = new FileWriter(infile.getAbsolutePath());
		FileWriter outwriter = new FileWriter(outfile.getAbsolutePath());

		Random r = new Random();
		for (int i = 0; i < size; i++) {
			inwriter.write(r.nextDouble() + " " + r.nextDouble() + "\n");
			outwriter.write(r.nextInt(2) + "\n");
		}
		inwriter.close();
		outwriter.close();
	}

	public static void trackCrossValidation(String filename, int reps, int epochsPerRep, int printFrequency,
			ANN ann, double[][] inputs, double[][] outputs) throws IOException {
		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		double[][] trainin = new double[inputs.length/2][inputs[0].length];
		double[][] trainout = new double[outputs.length/2][outputs[0].length];
		double[][] testin = new double[inputs.length-trainin.length][inputs[0].length];
		double[][] testout = new double[outputs.length-trainout.length][outputs[0].length];

		ANN.randomTrainPopulation(inputs, trainin, testin, outputs, trainout, testout);

		f.write("0\t");
		for (int i = 0; i < epochsPerRep; i++) {
			if (i%printFrequency == 0) f.write(i + "\t");
		}
		f.write("\n");
		for (int q = 0; q < reps; q++) {
			f.write(q + "\t");
			ann.reset();
			for (int i = 0; i < epochsPerRep; i++) {
				ANN.permute(trainin, trainout);
				for (int j = 0; j < trainin.length; j++) {
					Pattern trial = new Pattern(trainin[j], trainout[j]);
					ann.pass(trial);
					ann.updateAllWeights(trial);
				}
				double err = ANN.populationError(ann.outputs(testin), testout);
				if (i%printFrequency == 0) /*f.write(Arrays.deepToString(inputs)+ " " + Arrays.deepToString(ann.outputs(inputs))+"\t");//*/f.write(err + "\t");
			}
			f.write("\n");
		}

		f.close();
	}

	public static void compareMomentumHL(String filename, int reps, int epochsPerRep, int printFrequency,
			ANN ann, double[] mentums, int[] hlsizes, double[][] inputs, double[][] outputs) throws IOException {
		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		int numDataPoints = epochsPerRep/printFrequency + 1;

		f.write("0\t");
		for (int i = 0; i < epochsPerRep; i++) {
			if (i%printFrequency == 0) f.write(i + "\t");
		}
		f.write("\n");

		for (int mentum = 0; mentum < mentums.length; mentum++) {
			ann.momentum = mentums[mentum];
			System.out.println("mentum is " + ann.momentum);
			for (int hlsize = 0; hlsize  < hlsizes.length; hlsize++) {
				ann = new ANN(1, ann.learningSpeed, ann.momentum, ann.layerSize(0), hlsizes[hlsize], ann.layerSize(2));
				System.out.println("hlsize is " + ann.layerSize(1));

				double[][] tracks = new double[numDataPoints][reps];

				for (int q = 0; q < reps; q++) {
					ann.reset();

					double initerr = ANN.populationError(ann.outputs(inputs), outputs);
					tracks[0][q] = initerr;

					int dataPoint = 1;
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
							tracks[dataPoint][q] = err;
							dataPoint++;
						}
					}
				}
				f.write(mentums[mentum] + "/" + hlsizes[hlsize] + "\t");
				for (int dp = 0; dp < tracks.length; dp++) {
					f.write(median(tracks[dp]) + "\t");
				}
				f.write("\n");
			}
		}
		f.close();
	}

	public static void findBestNoiseForGen(String filename, int reps, 
			double maxError, ANN ann, double[] noises, double[][] inputs, double[][] outputs) throws IOException {
		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());
		double[][] trainin = new double[inputs.length/2][inputs[0].length];
		double[][] trainout = new double[outputs.length/2][outputs[0].length];
		double[][] testin = new double[inputs.length-trainin.length][inputs[0].length];
		double[][] testout = new double[outputs.length-trainout.length][outputs[0].length];

		ANN.randomTrainPopulation(inputs, trainin, testin, outputs, trainout, testout);

		//double[] noises = {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40};
		//0.50, 0.75, 0.95};//, 1.0};

		//double maxError = 0.1;
		for (int noise = 0; noise < noises.length; noise++) {
			double currnoise = noises[noise];
			double[] results = new double[reps];

			for (int rep = 0; rep < reps; rep++) {
				double err = 1.0;
				ann.reset();
				while (err > maxError) {
					ANN.permute(trainin, trainout);
					for (int j = 0; j < trainin.length; j++) {
						Pattern trial = new Pattern(trainin[j], trainout[j]);
						trial.setNoise(currnoise);
						ann.pass(trial);
						ann.updateAllWeights(trial);
					}
					err = ANN.populationError(ann.outputs(inputs), outputs);
				}

				double testPopError = ANN.populationError(ann.outputs(testin), testout);
				results[rep] = testPopError;

			}

			f.write(currnoise + "\t" + median(results) + "\t" + mean(results));
		}
		f.close();
	}

	public static void trackNoiseGeneralisation(String filename, int reps, int epochsPerRep, int printFrequency,
			double maxError, ANN ann, double[] noises, double[][] inputs, double[][] outputs) throws IOException {
		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		double[][] trainin = new double[inputs.length/2][inputs[0].length];
		double[][] trainout = new double[outputs.length/2][outputs[0].length];
		double[][] testin = new double[inputs.length-trainin.length][inputs[0].length];
		double[][] testout = new double[outputs.length-trainout.length][outputs[0].length];

		ANN.randomTrainPopulation(inputs, trainin, testin, outputs, trainout, testout);

		double[] o1 = {1.0, 0.0, 0.0};
		double[] o2 = {0.0, 1.0, 0.0};
		double[] o3 = {0.0, 0.0, 1.0};
		double[][] outputcats = {o1, o2, o3};
		String[] outputcatstring = {"1", "2", "3"};
		f.write("0\t");

		int[] dist = new int[outputcats.length];

		double[][] tempin = new double[testin.length][testin[0].length];
		double[][] tempout = new double[testout.length][testout[0].length];
		int index = 0;
		for (int cat = 0; cat < outputcats.length; cat++) {
			for (int test = 0; test < testin.length; test++) {
				if (Arrays.equals(testout[test], outputcats[cat])) {
					tempin[index] = testin[test];
					tempout[index] = testout[test];
					index++;
					dist[cat]++;
				}
			}
		}

		testin = tempin;
		testout = tempout;

		for (int cat = 0; cat < dist.length; cat++) {
			for (int noise = 0; noise < noises.length; noise++) {
				for (int d = 0; d < dist[cat]; d++){
					f.write(outputcatstring[cat] + "." + (noise+1) + "\t");
				}
			}
		}

		//System.err.println(Arrays.toString(dist));
		f.write("\n");

		for (int noise = 0; noise < noises.length; noise++) {
			ann.reset();
			double currnoise = noises[noise];
			double err = 1.0;
			int traincounter = 0;
			while (err > maxError) {
				traincounter++;
				ANN.permute(trainin, trainout);
				for (int j = 0; j < trainin.length; j++) {
					Pattern trial = new Pattern(trainin[j], trainout[j]);
					trial.setNoise(currnoise);
					ann.pass(trial);
					ann.updateAllWeights(trial);
				}
				err = ANN.populationError(ann.outputs(inputs), outputs);
			}
			System.err.println("Noise: " + currnoise + " trained for " + traincounter + " epochs.");

			double[] errors = new double[testin.length];

			for (int test = 0; test < testin.length; test++) {
				ann.pass(new Pattern(testin[test], null));
				errors[test] = ANN.patternError(ann.getCurrentOutput(), testout[test]);
			}

			DecimalFormat df = new DecimalFormat("0.00");

			f.write(currnoise + " MD: " + df.format(median(errors)) + 
					" ME: " + df.format(mean(errors)) + "\t");


			int counter = 0;
			for (int c = 0; c < dist.length; c++) {
				for (int n = 0; n < noises.length; n++) {
					if (n == noise) {
						for (int d = 0; d < dist[c]; d++) {
							f.write(errors[counter++] + "\t");
						}
					} else {
						for (int d = 0; d < dist[c]; d++) {
							f.write("\t");
						}

					}
				}
			}

			f.write("\n");

		}

		f.close();
	}

	public static void compareHiddenLayerSizes(String filename, int reps, int epochsPerRep, int printFrequency,
			ANN ann, int[] hlsizes, double[][] inputs, double[][] outputs) throws IOException {
		//ANN ann = new ANN(1, 0.5, 0.5, 2, 2, 1);
		int numDataPoints = epochsPerRep/printFrequency;
		int mloc = reps/2;
		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());
		f.write("0\t");
		for (int i = 0; i < epochsPerRep; i++) {
			if (i%printFrequency == 0) f.write(i + "\t");
		}
		f.write("\n");

		double[][] medianEpochPerMentum = new double[hlsizes.length][numDataPoints];

		for (int size = 0; size < hlsizes.length; size++) {
			// ann = new ANN(1, 0.5, 0.5, 2, hlsizes[size], 1);
			ann = new ANN(1, 0.5, 0.5, 4, hlsizes[size], 3);
			System.out.println("hlsize is " + hlsizes[size]);
			double[][] tracks = new double[numDataPoints][reps];
			for (int q = 0; q < reps; q++) {
				ann.reset();
				int dataPoint = 0;
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
						tracks[dataPoint][q] = err;
						dataPoint++;
					}
				}
			}
			f.write(hlsizes[size] + "\t");
			for (int dp = 0; dp < tracks.length; dp++) {
				Arrays.sort(tracks[dp]);
				medianEpochPerMentum[size][dp] = tracks[dp][mloc];
				f.write(tracks[dp][mloc] + "\t");
			}
			f.write("\n");
		}
		f.close();
	}

	public static void compareMomentumLearningSpeed(String filename, int reps, int epochsPerRep, int printFrequency,
			ANN ann, double[] mentums, double[] lisps, double[][] inputs, double[][] outputs) throws IOException {
		int numDataPoints = epochsPerRep/printFrequency;
		int mloc = reps/2;
		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());
		f.write("0\t");
		for (int i = 0; i < epochsPerRep; i++) {
			if (i%printFrequency == 0) f.write(i + "\t");
		}
		f.write("\n");

		double[][] medianEpochPerML = new double[mentums.length*lisps.length][numDataPoints];

		for (int mentum = 0; mentum < mentums.length; mentum++) {
			ann.momentum = mentums[mentum];
			System.out.println("mentum is " + ann.momentum);
			for (int lisp = 0; lisp  < lisps.length; lisp++) {
				ann.learningSpeed = lisps[lisp];
				System.out.println("lisp is " + ann.learningSpeed);

				double[][] tracks = new double[numDataPoints][reps];

				for (int q = 0; q < reps; q++) {
					ann.reset();
					int dataPoint = 0;
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
							tracks[dataPoint][q] = err;
							dataPoint++;
						}
					}
				}
				f.write("M" + mentums[mentum] + "/L" + lisps[lisp] + "\t");
				for (int dp = 0; dp < tracks.length; dp++) {
					Arrays.sort(tracks[dp]);
					medianEpochPerML[mentum*lisps.length + lisp][dp] = tracks[dp][mloc];
					f.write(tracks[dp][mloc] + "\t");
				}
				f.write("\n");
			}
		}
		f.close();
	}

	public static void findBestMomentum(String filename, int reps, int epochsPerRep, int printFrequency,
			ANN ann, double[] mentums, double[][] inputs, double[][] outputs) throws IOException {
		//ANN ann = new ANN(1, 0.5, 0.1, 2, 3, 1);

		int numDataPoints = epochsPerRep/printFrequency;
		int mloc = reps/2;
		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());
		f.write("0\t");
		for (int i = 0; i < epochsPerRep; i++) {
			if (i%printFrequency == 0) f.write(i + "\t");
		}
		f.write("\n");
		//double[] mentums = {0.0, 0.10, 0.20, 0.30, 0.40,
		//	0.50, 0.60, 0.70, 0.80, 0.90};//, 1.0}; // 0.9x had no effect (true)

		double[][] medianEpochPerMentum = new double[mentums.length][numDataPoints];

		for (int mentum = 0; mentum < mentums.length; mentum++) {
			ann.momentum = mentums[mentum];
			System.out.println("mentum is " + ann.momentum);
			double[][] tracks = new double[numDataPoints][reps];

			for (int q = 0; q < reps; q++) {
				ann.reset();
				int dataPoint = 0;
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
						tracks[dataPoint][q] = err;
						dataPoint++;
					}
				}
			}
			f.write(mentums[mentum] + "\t");
			for (int dp = 0; dp < tracks.length; dp++) {
				Arrays.sort(tracks[dp]);
				medianEpochPerMentum[mentum][dp] = tracks[dp][mloc];
				f.write(tracks[dp][mloc] + "\t");
			}
			f.write("\n");
		}
		f.close();
		// for each momentum
		//	run 50 trials
		//	take the median time at each epoch
		// output median times at each epoch

	}

	public static void localMinPointsXOR(String filename, int reps, int epochsPerRep, double maxError,
			ANN ann, double[][] inputs, double[][] outputs) throws IOException {

		double[][] pinputs = Arrays.copyOf(inputs, inputs.length);
		double[][] poutputs = Arrays.copyOf(outputs, outputs.length);
		File file = new File(filename); //new File("omg_ann_or.txt");
		FileWriter f = new FileWriter(file.getAbsolutePath());
		f.write("0\t");
		for (int i = 1; i <= outputs.length; i++) {
			f.write(i + "\t");
		}
		for (int i = 1; i <= outputs.length; i++) {
			f.write(i + "\t");
		}
		f.write("\n");
		for (int q = 0; q < reps; q++) {
			ann.reset();
			for (int i = 0; i < epochsPerRep; i++) {
				ANN.permute(pinputs, poutputs);
				for (int j = 0; j < inputs.length; j++) {
					Pattern trial = new Pattern(inputs[j], outputs[j]);
					ann.pass(trial);
					ann.updateAllWeights(trial);
				}
			}
			double[][] outs = ann.outputs(inputs);
			double err = ANN.populationError(outs, outputs);
			if (err > maxError) {
				if (err > 0.075) {
					f.write("MIN1\t");
					for (double[] out : outs) {
						f.write("\t");
					}
				} else {
					f.write("MIN2\t");
				}
				for (double[] out : outs) {
					f.write(out[0] + "\t");
				}
				f.write("\n");
			}
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
				if (i%printFrequency == 0) /*f.write(Arrays.deepToString(inputs)+ " " + Arrays.deepToString(ann.outputs(inputs))+"\t");//*/f.write(err + "\t");
			}
			f.write("\n");
		}
		f.close();
	}

	public static double mean(double[] nums) {
		double sum = 0;
		for (int i = 0; i < nums.length; i++) {
			sum += nums[i];
		}
		return sum / nums.length;
	}

	public static double median(double[] nums) {
		double[] sorted = Arrays.copyOf(nums, nums.length);
		Arrays.sort(sorted);
		return sorted[sorted.length/2];
	}

}
