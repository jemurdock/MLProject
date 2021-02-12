

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

public class Perceptron {	
	
	public Perceptron(){
		
	}

	public Result run(ArrayList<HashMap<Integer, Double>> data, double r, int e){
		//Initialize weights,bias term, and result object
		Random rng = new Random();
		rng.setSeed(12);
		int maxFeats = 0;
		for(int i = 0; i < data.size(); i++){
			int max = Collections.max(data.get(i).keySet());
			if(max > maxFeats)
				maxFeats = max;
		}
	    Double[] w = new Double[maxFeats];
	    for(int i = 0; i < w.length; i++){
	    	w[i] = rng.nextDouble();
	    	while(w[i] == 0)
	    		w[i] = rng.nextDouble();
	    }
	    double b = rng.nextDouble();
	    while(b == 0)
	    	b = rng.nextDouble();
	    Result result = new Result(w, 0, 0, r, -1);
	    double accuracy = 0;
	    
	    //For each epoch
	    for(int j = 0; j < e; j++){
	    	double mistakes = 0.0;
	    	Collections.shuffle(data);
	    	//For each example, predict labels.
	    	for(int k = 0; k < data.size(); k++){
	    		HashMap<Integer, Double> ex = data.get(k);
	    		double label = predict(w, b, ex);
	    		//System.out.println(label+", "+ex.get(0));
	    		//If a mistake was made, update the weights.
	    		if((label < 0 && ex.get(0) > 0) || (label > 0 && ex.get(0) < 0)){
	    			mistakes++;
	    			w = update(w, r, ex);
	    			b += r*ex.get(0);
	    			//System.out.println(w);
	    		}
	    	}
	    	//Keep track of the most accurate epoch so far.
	    	double acc = 1-(mistakes/data.size());
	    	if(acc > accuracy){
	    		accuracy = acc;
	    		result = new Result(w, acc, mistakes, r, -1);
	    	}
	    }
	    return result;
	}
	
	public double predict(Double[] w, double b, HashMap<Integer, Double> example){
		double prediction = 0;
		for(int i = 1; i < example.size(); i++){
			if(example.get(i) != null)
				prediction += w[i-1]*example.get(i);
		}
		return prediction += b;
	}

	public Double[] update(Double[] w, double r, HashMap<Integer, Double> example){
		for(int i = 1; i < example.size(); i++){
			if(example.get(i) != null){
				double product = example.get(0)*example.get(i)*r;
				w[i-1] += product;
			}
			
		}
		return w;
	}
	
	public double test(ArrayList<HashMap<Integer, Double>> testset, Double[] w, double b){
		double mistakes = 0;
		for(int i = 0; i < testset.size(); i++){
			HashMap<Integer, Double> ex = testset.get(i);
			double prediction = predict(w, b, ex);
			if((prediction < 0 && ex.get(0) > 0) || (prediction > 0 && ex.get(0) < 0))
				mistakes += 1;
		}
		return 1-(mistakes/testset.size());
	}
	
	public int[] eval(ArrayList<HashMap<Integer, Double>> evalset, Double[] w, double b){
		int[] predictions = new int[evalset.size()];
		for(int i = 0; i < evalset.size(); i++){
			int label = (int)predict(w, b, evalset.get(i));
			predictions[i] = label > 0 ? 1:0;
		}
		return predictions;		
	}
}
