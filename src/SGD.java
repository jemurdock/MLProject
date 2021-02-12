

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

public class SGD {

	public double r;
	public double c;
	public double e;
	public int f;
	
	SGD(double r, double c, double e, int f){
		this.r = r;
		this.c = c;
		this.e = e;
		this.f = f;
	}
	
	public Result run(ArrayList<HashMap<Integer, Double>> data, Double[] w){		
	    Result result = new Result(w, 0, 0, this.r, this.c);
	    double accuracy = 0;
	    Random rng = new Random(12); //Keep shuffling consistent
	    for(int j = 0; j < this.e; j++){
	    	double mistakes = 0;
	    	Collections.shuffle(data, rng);
	    	double rate = this.r/(1.0+j);
	    	for(int k = 0; k < data.size(); k++){
	    		double value = predict(w, data.get(k));
    			for(int i = 0; i < w.length; i++){
    				w[i] = (1-rate)*w[i];
    			}   			
	    		if(value <= 1){
	    			w = update(w, rate, data.get(k));
	    			mistakes++;
	    		}
	    	}
	    	//Keep track of the most accurate epoch so far.
	    	double acc = 1-(mistakes/data.size());
	    	if(acc > accuracy){
	    		accuracy = acc;
	    		result = new Result(w.clone(), acc, mistakes, this.r, this.c);
	    	}
	    }
		return result;
	}
	
	public double predict(Double[] w, HashMap<Integer, Double>example){
		double prediction = 0;
		for(int i = 0; i < w.length-1; i++){
			if(example.get(i) != null)
				prediction += w[i]*example.get(i);
		}
		prediction += w[w.length-1];
		return prediction * example.get(-1);
	}

	public Double[] update(Double[] w, double rate, HashMap<Integer, Double> example){
		for(int i = 0; i < w.length-1; i++){
			if(example.get(i) != null)
				w[i] += example.get(-1)*this.c*rate*example.get(i);			
		}
		w[w.length-1] = (1-rate)*w[w.length-1]+example.get(-1)*this.c*rate;
		return w;
	}
	
	public double test(ArrayList<HashMap<Integer, Double>> testset, Double[] w){
		double mistakes = 0;
		for(int i = 0; i < testset.size(); i++){
			HashMap<Integer, Double> ex = testset.get(i);
			double prediction = predict(w, ex);
			if(prediction <= 1)
				mistakes += 1;
		}
		return 1-(mistakes/testset.size());
	}	
	
	public int[] eval(ArrayList<HashMap<Integer, Double>> evalset, Double[] w){
		int[] predictions = new int[evalset.size()];
		for(int i = 0; i < evalset.size(); i++){
			double label = predict(w, evalset.get(i));
			predictions[i] = label <= 1 ? 0:1;
		}
		return predictions;
	}
	
}
