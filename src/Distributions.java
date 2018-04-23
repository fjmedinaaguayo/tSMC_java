import java.util.*;
import java.util.stream.IntStream;

import org.apache.commons.math3.distribution.*;
import org.apache.commons.math3.util.CombinatoricsUtils;

interface Distribution {

    double logDensity(List<Object> x);
    List<ArrayList<Object>> Draw(int N);
}

abstract class Kernel {

    abstract double logDensity(List<Object> x, List<Object> x0);
    abstract List<ArrayList<Object>> Draw(List<ArrayList<Object>> x0);
}

class MCMCK extends Kernel {

    private TargetDistn pi;
    private Kernel q;

    MCMCK(TargetDistn pi_, Kernel q_) {

        this.pi=pi_;
        this.q=q_;
    }

    List<ArrayList<Object>> Draw(List<ArrayList<Object>> x0){

        int moveInd=1;
        if(moveInd==0)
            return x0;

        List<ArrayList<Object>> x=q.Draw(x0);
        int xSize=x0.size();

        ArrayList<Double> logRatio=new ArrayList(Collections.nCopies(xSize, 0));
        ArrayList<Integer> accept=new ArrayList(Collections.nCopies(xSize, 0));
        IntStream.range(0, x0.size()).//parallel().
                forEach(i -> {

            double logRatioDen = pi.logDensity(x0.get(i)) + q.logDensity(x.get(i), x0.get(i));
            double logRatioNum = pi.logDensity(x.get(i)) + q.logDensity(x0.get(i), x.get(i));

            if(logRatioNum==-1*Double.POSITIVE_INFINITY)
                logRatio.set(i,-1*Double.POSITIVE_INFINITY);
            else
                logRatio.set(i,logRatioNum-logRatioDen);

            if(Double.isNaN(logRatio.get(i))){
                logRatio.set(i, pi.logDensity(x.get(i)) + q.logDensity(x0.get(i), x.get(i))
                        -pi.logDensity(x0.get(i)) - q.logDensity(x.get(i), x0.get(i)) );

                logRatio.set(i,-1*Double.POSITIVE_INFINITY);
            }

            if(Math.log(Math.random())>logRatio.get(i))
                x.set(i,(ArrayList) x0.get(i));
            else
                accept.set(i,1);

        });

        System.out.print("MCMC accept: "+Utils.mean(accept)+", ");
        return x;
    }

    double logDensity(List<Object> x, List<Object> x0){
        return 0.0;
    }
}

class aGRWMetropK extends Kernel {

    private TargetDistn pi;
    private Kernel q;

    aGRWMetropK(TargetDistn pi_, Kernel q_) {

        this.pi=pi_;
        this.q=q_;
    }

    public void setKernel(Kernel q_) {

        this.q=q_;
    }

    List<ArrayList<Object>> Draw(List<ArrayList<Object>> x0){

        int moveInd=1;
        double iters=0, itersMax=1e+1;
        if(moveInd==0)
            return x0;

        double mean_accept=0;
        List<Object> y=new ArrayList();
        y.add(x0);

        //while(mean_accept<0.08 || mean_accept>0.12){
        while(iters<itersMax){

            final List<ArrayList<Object>> x_temp=(ArrayList) y.get((int) iters);
            List<ArrayList<Object>> x=q.Draw(x_temp);
            int xSize=x0.size();

            ArrayList<Double> logRatio=new ArrayList(Collections.nCopies(xSize, 0));
            ArrayList<Integer> accept=new ArrayList(Collections.nCopies(xSize, 0));
            IntStream.range(0, x0.size()).parallel().
                    forEach(i -> {

                double logRatioDen = pi.logDensity(x_temp.get(i)) + q.logDensity(x.get(i), x_temp.get(i));
                double logRatioNum = pi.logDensity(x.get(i)) + q.logDensity(x_temp.get(i), x.get(i));

                if(logRatioNum==-1*Double.POSITIVE_INFINITY)
                    logRatio.set(i,-1*Double.POSITIVE_INFINITY);
                else
                    logRatio.set(i,logRatioNum-logRatioDen);

                if(Double.isNaN(logRatio.get(i))){
                    logRatio.set(i, pi.logDensity(x.get(i)) + q.logDensity(x_temp.get(i), x.get(i))
                            -pi.logDensity(x_temp.get(i)) - q.logDensity(x.get(i), x_temp.get(i)) );

                    //logRatio.set(i,-1*Double.POSITIVE_INFINITY);
                }

                if(Math.log(Math.random())>logRatio.get(i))
                    x.set(i,(ArrayList) x_temp.get(i));
                else
                    accept.set(i,1);

            });

            iters+=1;
            mean_accept+=Utils.mean(accept)/itersMax;
            //System.out.print("MCMC accept: "+mean_accept+", ");

            //((Proposal3GaussTLogit2) q).setPropVarFactor(Math.max(mean_accept,1e-5)/0.1);

            y.add(x);
        }
        System.out.print("MCMC accept: "+mean_accept+", ");

        //WriteFile.writeMCMC("MCMC_"+(int) itersMax,y);
        return (List<ArrayList<Object>>) y.get((int) itersMax);
    }

    double logDensity(List<Object> x, List<Object> x0){
        return 0.0;
    }
}

@FunctionalInterface
interface TargetDistn {

    double logDensity(List<Object> x);
}

class GaussianK extends Kernel {

    private double[][] S2;

    GaussianK(double[][] S2_) {

        this.S2=S2_;
    }

    public double logDensity(List<Object> x, List<Object> x0) {

        double[] x0_=Utils.convertDoubles(x0);

        Distribution d=new GaussianD(x0_,S2);

        return d.logDensity(x);
    }

    public List<ArrayList<Object>> Draw(List<ArrayList<Object>> x0) {

        List<ArrayList<Object>> x=new ArrayList(x0.size());

        for(int i=0; i<x0.size(); i++){

            double[] x0_=Utils.convertDoubles(x0.get(i));
            Distribution d=new GaussianD(x0_,S2);
            x.add(i,d.Draw(1).get(0));
        }

        return x;
    }
}

class GaussianD implements Distribution, TargetDistn {

    private MultivariateNormalDistribution MVN;

    GaussianD(double[] mu, double[][] S2) {

        MVN=new MultivariateNormalDistribution(mu, S2);
    }

    public double logDensity(List<Object> x) {

        double[] x_=Utils.convertDoubles(x);

        return Math.log(MVN.density(x_));
    }

    public List<ArrayList<Object>> Draw(int N) {

        List samp = new ArrayList<Double[]>(N);

        double[][] temp=MVN.sample(N);

        List<double[]> list=Arrays.asList(temp);
        for(double[] array : list){
            samp.add( Utils.convertDoubles(array) );
        }

        return samp;
    }
}

class tGaussianD implements Distribution{

    private double mu, s2;
    private double lb, ub;

    tGaussianD(double mu_, double s2_, double lb_, double ub_) {

        mu=mu_;
        s2=s2_;
        lb=lb_;
        ub=ub_;
    }

    public double logDensity(List<Object> x) {

        double[] x_=Utils.convertDoubles(x);
        NormalDistribution N=new NormalDistribution(mu, Math.sqrt(s2));

        double Z=N.cumulativeProbability(ub)-N.cumulativeProbability(lb);

        return N.logDensity(x_[0])-Math.log(Z);
    }

    public List<ArrayList<Object>> Draw(int N) {

        List samp = new ArrayList<Double[]>(N);
        NormalDistribution Norm=new NormalDistribution(mu, Math.sqrt(s2));

        for(int i=0; i<N; i++){

            double z=Norm.sample();
            int count=0;
            while( (z<=lb) || (z>=ub) ){
                z=Norm.sample();
                count++;
            }

            List temp2=new ArrayList<Double>();
            temp2.add(z);
            samp.add( temp2 );
        }

        return samp;
    }
}

class GammaD implements Distribution {

    private List<GammaDistribution> Gamma;

    GammaD(double[] shape, double[] rate){

        int d=shape.length;
        List<GammaDistribution> Gamma=new ArrayList<>(d);

        for(int i=0; i<d; i++){

            GammaDistribution temp=new GammaDistribution(shape[i], 1.0/rate[i]);
            Gamma.add(temp);
        }

        this.Gamma=Gamma;
    }

    public double logDensity(List<Object> x) {

        double[] x_=Utils.convertDoubles(x);

        return Gamma.get(0).logDensity(x_[0]);
    }

    public List<ArrayList<Object>> Draw(int N) {

        List<ArrayList<Object>> samp = new ArrayList<>(N);

        int d=Gamma.size();

        double[] temp=Gamma.get(0).sample(N);

        for(double value: temp){

            ArrayList temp2=new ArrayList<Double>();
            temp2.add(value);
            samp.add(temp2);
        }

        for(int i=1; i<d; i++){

            temp=Gamma.get(i).sample(N);

            for(int j=0; j<N; j++){

                samp.get(j).add(temp[j]);
            }
        }

        return samp;
    }
}

class Dirichlet1D implements Distribution {

    private int K;
    private GammaDistribution Gamma=new GammaDistribution(1,1);

    Dirichlet1D(int K) {

        this.K=K;
    }

    public double logDensity(List<Object> x) {

        return CombinatoricsUtils.factorialLog(K-1);
    }

    // Draw is returned as log.
    public List<ArrayList<Object>> Draw(int N) {

        List samp = new ArrayList<Double[]>(N);

        for(int i=0; i<N; i++){

            List<Double> temp=Utils.convertDoubles(Gamma.sample(K));
            temp=Utils.log(temp);

            temp=Utils.sum(temp,-Utils.logSumExp(temp));

            samp.add(temp);
        }

        return samp;
    }
}

class BetaD implements Distribution {

    private BetaDistribution Beta;

    BetaD(double alpha, double beta) {

        Beta=new BetaDistribution(alpha, beta);
    }

    public double logDensity(List<Object> x) {

        double[] x_=Utils.convertDoubles(x);

        return Beta.logDensity(x_[0]);
    }

    public List<ArrayList<Object>> Draw(int N) {

        List samp = new ArrayList<Double[]>();

        double[] temp=Beta.sample(N);

        for(double value: temp){

            List temp2=new ArrayList<Double>();
            temp2.add(0,value);
            samp.add( temp2 );
        }

        return samp;
    }
}

class dUniformD implements Distribution {

    private UniformIntegerDistribution Unif;

    dUniformD(int K_) {

        this.Unif=new UniformIntegerDistribution(1,K_);
    }

    public double logDensity(List<Object> x) {

        double[] x_=Utils.convertDoubles(x);

        return Unif.logProbability((int) x_[0]);
    }

    public List<ArrayList<Object>> Draw(int N) {

        List samp = new ArrayList<Integer[]>(N);

        int[] temp=Unif.sample(N);

        for(int value: temp){

            List temp2=new ArrayList<Integer>();
            temp2.add(0,value);
            samp.add( temp2 );
        }

        return samp;
    }
}

class BinomD implements Distribution {

    private BinomialDistribution Binom;

    BinomD(int K, double p) {

        this.Binom=new BinomialDistribution(K,p);
    }

    public double logDensity(List<Object> x) {

        double[] x_=Utils.convertDoubles(x);

        return Binom.logProbability((int) x_[0]);
    }

    public List<ArrayList<Object>> Draw(int N) {

        List samp = new ArrayList<Integer[]>(N);

        int[] temp=Binom.sample(N);

        for(int value: temp){

            List temp2=new ArrayList<Integer>();
            temp2.add(0,value);
            samp.add( temp2 );
        }

        return samp;
    }
}

class GaussGammaPrior implements Distribution {

    int M;
    double prior_mean, prior_sd;
    double prior_shape, prior_rate;

    GaussGammaPrior(int M_, double prior_mean_, double prior_sd_,
                    double prior_shape_, double prior_rate_) {

        this.M=M_;
        this.prior_mean=prior_mean_;
        this.prior_sd=prior_sd_;
        this.prior_shape=prior_shape_;
        this.prior_rate=prior_rate_;
    }

    public double logDensity(List<Object> x) {

        GammaDistribution GammaPrior=new GammaDistribution(prior_shape,1.0/prior_rate);
        NormalDistribution NormalPrior=new NormalDistribution(prior_mean,prior_sd);
        Dirichlet1D Dirichlet=new Dirichlet1D(M);
        List logPriorTerm=new ArrayList<Double>(M);

        double logConst=CombinatoricsUtils.factorialLog(M);

        TripletMixGauss triplet=new TripletMixGauss(x.subList(0,3*M-1));

        List logWeights=triplet.getLogWeightsFull();
        List Means=triplet.getMeans();
        List logPrecisions=triplet.getLogPrecisions();

        if(Utils.areOrdered(Means)==0 || triplet.areWeightsValid()==0){

            return -1*Double.POSITIVE_INFINITY;
        }

        for(int k=0; k<M; k++){

            logPriorTerm.add(NormalPrior.logDensity((double) Means.get(k))
                    +GammaPrior.logDensity(Math.exp((double) logPrecisions.get(k))) );
        }

        return logConst+Utils.sum(logPriorTerm)+Dirichlet.logDensity((List) Utils.exp(logWeights));
    }

    public List<ArrayList<Object>> Draw(int N) {

        Distribution GammaPrior=new GammaD(Utils.equalVector(prior_shape,M),
                Utils.equalVector(prior_rate,M));

        double[] mu=Utils.equalVector(prior_mean,M);
        double[][] S2=Utils.idMatrix(Math.pow(prior_sd,2),M);
        Distribution GaussPrior=new GaussianD(mu,S2);

        Distribution DirichPrior=new Dirichlet1D(M);

        List<ArrayList<Object>> sampleLogWeights=DirichPrior.Draw(N);
        List<ArrayList<Object>> sampleMeans=GaussPrior.Draw(N);
        List<ArrayList<Object>> samplePrecs=GammaPrior.Draw(N);
        List<ArrayList<Object>> sample=new ArrayList<>(N);

        for(int i=0; i<N; i++){

            List<Double> sampleMeans_i=Utils.toListDoubles(sampleMeans.get(i));
            Collections.sort(sampleMeans_i);

            TripletMixGauss triplet=new TripletMixGauss(Utils.toListDoubles(sampleLogWeights.get(i)),
                    sampleMeans_i, Utils.log(samplePrecs.get(i)));

            sample.add((ArrayList<Object>) triplet.asSingleVector());
        }

        return sample;
    }
}

class BirthProposal implements Distribution {

    int M;
    double prior_mean, prior_sd;
    double prior_shape, prior_rate;

    BirthProposal(int M, List params) {

        this.M=M;
        this.prior_mean=(double) params.get(0);
        this.prior_sd=(double) params.get(1);
        this.prior_shape=(double) params.get(2);
        this.prior_rate=(double) params.get(3);
    }

    public double logDensity(List<Object> x){

        GammaDistribution GammaPrior=new GammaDistribution(prior_shape,1.0/prior_rate);
        NormalDistribution MeanPrior=new NormalDistribution(prior_mean,prior_sd);
        double result=0;

        if(M>=1){

            BetaDistribution BetaPrior=new BetaDistribution(1,M);

            List logitWeights= new ArrayList<>(x.subList(0,1));
            List logWeights= Utils.LogitToLog(logitWeights);

            List Means= new ArrayList<>(x.subList(1,2));
            List logPrecisions= new ArrayList<>(x.subList(2,3));

            result=BetaPrior.logDensity(Math.exp((double) logWeights.get(0)))
                    +MeanPrior.logDensity((double) Means.get(0))
                    +GammaPrior.logDensity(Math.exp((double) logPrecisions.get(0)));
        }

        return result;
    }

    public List<ArrayList<Object>> Draw(int N) {

        List<ArrayList<Object>> sample=new ArrayList<>(N);

        if(M>=1){

            Distribution GammaPrior=new GammaD(Utils.equalVector(prior_shape,1),
                    Utils.equalVector(prior_rate,1));

            double[] mu={prior_mean};
            double[][] S2={{Math.pow(prior_sd,2)}};
            Distribution GaussPrior=new GaussianD(mu,S2);

            List<ArrayList<Object>> sampleMean=GaussPrior.Draw(N);
            List<ArrayList<Object>> samplePrec=GammaPrior.Draw(N);

            Distribution BetaPrior=new BetaD(1,M);
            sample=BetaPrior.Draw(N);

            for(int i=0; i<N; i++){

                sample.get(i).set(0,-1*Math.log(
                        Math.pow((double) sample.get(i).get(0),-1)-1.0 ));
                sample.get(i).add(sampleMean.get(i).get(0));
                sample.get(i).add(Math.log((double) samplePrec.get(i).get(0)));
            }

        }

        return sample;
    }
}

class SplitProposal implements Distribution {

    int M;
    double alpha1, beta1, alpha2, beta2, alpha3, beta3;

    SplitProposal(int M, List params) {

        this.M=M;
        this.alpha1=(double) params.get(0);
        this.beta1=(double) params.get(1);
        this.alpha2=(double) params.get(2);
        this.beta2=(double) params.get(3);
        this.alpha3=(double) params.get(4);
        this.beta3=(double) params.get(5);
    }

    public double logDensity(List<Object> x) {

        double result=0;

        if(M>=1){

            BetaDistribution BetaPrior1=new BetaDistribution(alpha1,beta1);
            BetaDistribution BetaPrior2=new BetaDistribution(alpha2,beta2);
            BetaDistribution BetaPrior3=new BetaDistribution(alpha3,beta3);
            UniformIntegerDistribution Unif=new UniformIntegerDistribution(1,M);

            result=BetaPrior1.logDensity((double) x.get(0))
                    +BetaPrior2.logDensity((double) x.get(1))
                    +BetaPrior3.logDensity((double) x.get(2))
                    +Unif.logProbability(M);
        }

        return result;
    }

    public List<ArrayList<Object>> Draw(int N) {

        List<ArrayList<Object>> sample=new ArrayList<>(N);

        if(M>=1){

            Distribution BetaPrior1=new BetaD(alpha1,beta1);
            Distribution BetaPrior2=new BetaD(alpha2,beta2);
            Distribution BetaPrior3=new BetaD(alpha3,beta3);
            Distribution Unif=new dUniformD(M);

            List<ArrayList<Object>> u2=BetaPrior2.Draw(N);
            List<ArrayList<Object>> u3=BetaPrior3.Draw(N);
            List<ArrayList<Object>> k=Unif.Draw(N);

            sample=BetaPrior1.Draw(N);

            for(int i=0; i<N; i++){

                sample.get(i).add(u2.get(i).get(0));
                sample.get(i).add(u3.get(i).get(0));
                sample.get(i).add(k.get(i).get(0));
            }
        }

        return sample;
    }
}

class SMCProposal implements Distribution {

    BirthProposal Birth;
    SplitProposal Split;
    double probBirth;

    SMCProposal(int M, double probBirth, List<ArrayList> params) {

        this.probBirth=probBirth;
        this.Birth =new BirthProposal(M, params.get(0));
        this.Split =new SplitProposal(M, params.get(1));
    }

    public double logDensity(List<Object> x) {

        double result=0;

        int moveType=(int) x.get(0);

        if(moveType==1){

            //The density of the Bernoulli is not included in order to obtain the correct weight.
            //result=Math.log(this.probBirth);
            result+=Birth.logDensity(x.subList(1,x.size()));
        }
        else{

            //result=Math.log(1-this.probBirth);
            result+=Split.logDensity(x.subList(1,x.size()));
        }

        return result;
    }

    public List<ArrayList<Object>> Draw(int N) {

        List<ArrayList<Object>> sample=new ArrayList<>(N);

        Distribution Binom=new BinomD(1,probBirth);
        sample=Binom.Draw(N);

        for(int i=0; i<N; i++){

            if((int) sample.get(i).get(0)==1){

                List<ArrayList<Object>> temp=this.Birth.Draw(1);

                if(temp.size()>0)
                    sample.get(i).addAll(temp.get(0));
            }
            else{

                List<ArrayList<Object>> temp=this.Split.Draw(1);

                if(temp.size()>0)
                    sample.get(i).addAll(temp.get(0));
            }
        }

        return sample;
    }
}

class GaussMixPosterior implements TargetDistn {

    int M;
    Distribution Prior;
    List<Double> Obs;

    GaussMixPosterior(int M, List<Double> Obs_,
                      double prior_mean_, double prior_sd_,
                      double prior_shape_, double prior_rate_) {

        this.M=M;
        this.Obs=Obs_;
        Prior=new GaussGammaPrior(M,prior_mean_,prior_sd_,prior_shape_,prior_rate_);
    }

    public double logDensity(List<Object> x_) {

        double logPrior=Prior.logDensity(x_);

        if(logPrior==-1* Double.POSITIVE_INFINITY)
            return logPrior;

        int n=Obs.size();

        TripletMixGauss triplet=new TripletMixGauss(x_.subList(0,3*M-1));

        List logWeights=triplet.getLogWeightsFull();
        List Means=triplet.getMeans();
        List logPrecisions=triplet.getLogPrecisions();

        List<NormalDistribution> VectNormals=new ArrayList<>(M);

        for(int i=0; i<M; i++) {

            VectNormals.add(new NormalDistribution((double) Means.get(i),
                    Math.sqrt(Math.exp(-1 * (double) logPrecisions.get(i)))));
        }

        ArrayList<Double> tempLike = new ArrayList<>(n);

        for(int k=0; k<n; k++){

            ArrayList<Double> vecLike = new ArrayList<>(M);

            for(int i=0; i<M; i++){

                vecLike.add( ((double) logWeights.get(i)) + VectNormals.get(i).logDensity(Obs.get(k)));
            }

            tempLike.add(Utils.logSumExp(vecLike));

        }

        return Utils.sum(tempLike)+logPrior;
    }
}

class Proposal3Gauss extends Kernel{

    private double[][] S2_logitW;
    private double[][] S2_Means;
    private double[][] S2_logPrecs;

    Proposal3Gauss(double s2_logitW, double s2_Means, double s2_logPrecs, int d) {

        S2_logitW=Utils.idMatrix(s2_logitW/(3*d-1),d-1);
        S2_Means=Utils.idMatrix(s2_Means/(3*d-1),d);
        S2_logPrecs=Utils.idMatrix(s2_logPrecs/(3*d-1),d);
    }

    public double logDensity(List<Object> x, List<Object> x0) {

        double logDetJacobn=0, logDens=0;

        TripletMixGauss triplet=new TripletMixGauss(x.subList(0,x.size()-1));
        TripletMixGauss triplet0=new TripletMixGauss(x0.subList(0,x0.size()-1));

        double[] x0_Means=Utils.convertDoubles(triplet0.getMeans());
        double[] x0_logPrecs=Utils.convertDoubles(triplet0.getLogPrecisions());

        Distribution q_Means=new GaussianD(x0_Means,S2_Means);
        Distribution q_logPrecs=new GaussianD(x0_logPrecs,S2_logPrecs);

        List Means=triplet.getMeans();
        List logPrecisions=triplet.getLogPrecisions();

        int M=x.size()/3;

        if(M==1){

            logDetJacobn=-1*((double) logPrecisions.get(M-1));
            logDens=q_Means.logDensity(Means)+q_logPrecs.logDensity(logPrecisions);
        }
        else{

            double[] x0_logitW=Utils.convertDoubles(triplet0.getLogitWeights());

            List logitWeights=triplet.getLogitWeights();
            List logWeights=triplet.getLogWeightsFull();

            Distribution q_logitW=new GaussianD(x0_logitW,S2_logitW);

            logDens=q_logitW.logDensity(logitWeights)+q_Means.logDensity(Means)+q_logPrecs.logDensity(logPrecisions);

            for(int i=0; i<M-1; i++) {

                logDetJacobn+=-1*((double) logWeights.get(i)
                        +Math.log(1.0-Math.exp((double) logWeights.get(i)))
                        +(double) logPrecisions.get(i));
            }
            logDetJacobn+=-1*((double) logPrecisions.get(M-1));
        }

        return logDens+logDetJacobn;
    }

    public List<ArrayList<Object>> Draw(List<ArrayList<Object>> x0) {

        List<ArrayList<Object>> x=new ArrayList(x0.size());

        for(int i=0; i<x0.size(); i++){

            int M=(x0.get(i).size()+1)/3;

            if(M==1){

                double[] x0_Means=Utils.convertDoubles(x0.get(i).subList(0,1));
                double[] x0_logPrecs=Utils.convertDoubles(x0.get(i).subList(1,2));

                Distribution q_Means=new GaussianD(x0_Means,S2_Means);
                Distribution q_logPrecs=new GaussianD(x0_logPrecs,S2_logPrecs);

                ArrayList temp=new ArrayList<>(q_Means.Draw(1).get(0));
                temp.addAll(q_logPrecs.Draw(1).get(0));

                temp.addAll(x0.get(i).subList(2,3));

                x.add(i,temp);
            }
            else{

                double[] x0_logitW=Utils.convertDoubles(x0.get(i).subList(0,M-1));
                double[] x0_Means=Utils.convertDoubles(x0.get(i).subList(M-1,2*M-1));
                double[] x0_logPrecs=Utils.convertDoubles(x0.get(i).subList(2*M-1,3*M-1));

                Distribution q_logitW=new GaussianD(x0_logitW,S2_logitW);
                Distribution q_Means=new GaussianD(x0_Means,S2_Means);
                Distribution q_logPrecs=new GaussianD(x0_logPrecs,S2_logPrecs);

                ArrayList<Object> temp=new ArrayList<>(q_logitW.Draw(1).get(0));
                temp.addAll(q_Means.Draw(1).get(0));
                temp.addAll(q_logPrecs.Draw(1).get(0));

                temp.addAll(x0.get(i).subList(3*M-1,3*M));

                x.add(i,temp);
            }
        }

        return x;
    }
}

class Proposal3GaussLogit2 extends Kernel{

    private double[][] S2_logitW;
    private double[][] S2_Means;
    private double[][] S2_logPrecs;
    private double[][] S2_full;

    Proposal3GaussLogit2(double s2_logitW, double s2_Means, double s2_logPrecs, int d) {

        S2_logitW=Utils.idMatrix(s2_logitW/(3*d-1),d-1);
        S2_Means=Utils.idMatrix(s2_Means/(3*d-1),d);
        S2_logPrecs=Utils.idMatrix(s2_logPrecs/(3*d-1),d);

        S2_full=ReadFile.readMatrix("Sigma_MCMC.txt");
    }

    Proposal3GaussLogit2(double[][] Sigma) {

        int d=Sigma[0].length;
        S2_full=Utils.prod(Sigma,1.0/d);
    }

    public void setPropVarFactor(double factor){

        this.S2_logitW=Utils.prod(this.S2_logitW,factor);
        this.S2_Means=Utils.prod(this.S2_Means,factor);
        this.S2_logPrecs=Utils.prod(this.S2_logPrecs,factor);
    }

    public double logDensity(List<Object> x, List<Object> x0) {

        double logDetJacobn, logDens;

        TripletMixGauss triplet=new TripletMixGauss(x.subList(0,x.size()-1));
        TripletMixGauss triplet0=new TripletMixGauss(x0.subList(0,x0.size()-1));

        double[] mean0_MVN=Utils.convertDoubles(triplet0.asSingleVectorLogit2());

        Distribution q_MVN=new GaussianD(mean0_MVN,S2_full);

        List logWeights=triplet.getLogWeightsFull();
        List logPrecisions=triplet.getLogPrecisions();

        logDens=q_MVN.logDensity(triplet.asSingleVectorLogit2());
        logDetJacobn=-Utils.sum((List<Double>) logPrecisions)-Utils.sum((List<Double>) logWeights);

        return logDens+logDetJacobn;
    }

    public List<ArrayList<Object>> Draw(List<ArrayList<Object>> x0) {

        List<ArrayList<Object>> x=new ArrayList(x0.size());

        for(int i=0; i<x0.size(); i++){

            int M=(x0.get(i).size()+1)/3;

            TripletMixGauss triplet0=new TripletMixGauss(x0.get(i).subList(0,x0.get(i).size()-1));

            double[] mean0_MVN=Utils.convertDoubles(triplet0.asSingleVectorLogit2());

            Distribution q_MVN=new GaussianD(mean0_MVN,S2_full);

            TripletMixGauss triplet=new TripletMixGauss(q_MVN.Draw(1).get(0),"Logit2");
            ArrayList temp=new ArrayList<>(triplet.asSingleVector());
            temp.addAll(x0.get(i).subList(3*M-1,3*M));

            x.add(i,temp);
        }

        return x;
    }
}

class Proposal3GaussTLogit2 extends Kernel{

    private double[][] S2_logitW;
    private double[][] S2_Means;
    private double[][] S2_logPrecs;

    Proposal3GaussTLogit2(double s2_logitW, double s2_Means, double s2_logPrecs, int d) {

        S2_logitW=Utils.idMatrix(s2_logitW/(3*d-1),d-1);
        S2_Means=Utils.idMatrix(s2_Means/(3*d-1),d);
        S2_logPrecs=Utils.idMatrix(s2_logPrecs/(3*d-1),d);
    }

    public void setPropVarFactor(double factor){

        this.S2_logitW=Utils.prod(this.S2_logitW,factor);
        this.S2_Means=Utils.prod(this.S2_Means,factor);
        this.S2_logPrecs=Utils.prod(this.S2_logPrecs,factor);
    }

    public double logDensity(List<Object> x, List<Object> x0) {

        double logDetJacobn=0, logDens=0;

        TripletMixGauss triplet=new TripletMixGauss(x.subList(0,x.size()-1));
        TripletMixGauss triplet0=new TripletMixGauss(x0.subList(0,x0.size()-1));

        double[] x0_Means=Utils.convertDoubles(triplet0.getMeans());
        double[] x0_logPrecs=Utils.convertDoubles(triplet0.getLogPrecisions());

        Distribution q_Means=new GaussianD(x0_Means,S2_Means);
        Distribution q_logPrecs=new GaussianD(x0_logPrecs,S2_logPrecs);

        List Means=triplet.getMeans();
        List logPrecisions=triplet.getLogPrecisions();

        int M=x.size()/3;

        if(M==1){

            logDetJacobn=-1*((double) logPrecisions.get(M-1));
            logDens=q_Means.logDensity(Means)+q_logPrecs.logDensity(logPrecisions);
        }
        else{

            double[] x0_logit2W=Utils.convertDoubles(triplet0.getLogit2Weights());

            List logit2Weights=triplet.getLogit2Weights();
            List logWeights=triplet.getLogWeightsFull();

            Distribution q_logit2W=new GaussianD(x0_logit2W,S2_logitW);
            logDens=q_logit2W.logDensity(logit2Weights);

            for(int i=0; i<x0_Means.length; i++){

                Distribution q_Means_i;

                if(i==0)
                    q_Means_i=new tGaussianD(x0_Means[i],S2_Means[i][i],-1*Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
                else{

                    double mu=Math.max(x0_Means[i],(double) Means.get(i-1));

                    q_Means_i=new tGaussianD(mu,S2_Means[i][i],(double) Means.get(i-1), Double.POSITIVE_INFINITY);
                }

                logDens+=q_Means_i.logDensity(Means.subList(i,i+1));
            }

            logDens+=q_logPrecs.logDensity(logPrecisions);

            logDetJacobn=-Utils.sum((List<Double>) logPrecisions)-Utils.sum((List<Double>) logWeights);
        }

        return logDens+logDetJacobn;
    }

    public List<ArrayList<Object>> Draw(List<ArrayList<Object>> x0) {

        List<ArrayList<Object>> x=new ArrayList(x0.size());

        for(int i=0; i<x0.size(); i++){

            int M=(x0.get(i).size())/3;

            if(M==1){

                double[] x0_Means=Utils.convertDoubles(x0.get(i).subList(0,1));
                double[] x0_logPrecs=Utils.convertDoubles(x0.get(i).subList(1,2));

                Distribution q_Means=new GaussianD(x0_Means,S2_Means);
                Distribution q_logPrecs=new GaussianD(x0_logPrecs,S2_logPrecs);

                ArrayList temp=new ArrayList<>(q_Means.Draw(1).get(0));
                temp.addAll(q_logPrecs.Draw(1).get(0));

                temp.addAll(x0.get(i).subList(2,3));

                x.add(i,temp);
            }
            else{

                TripletMixGauss triplet0=new TripletMixGauss(x0.get(i).subList(0,3*M-1));

                double[] x0_logit2W=Utils.convertDoubles(triplet0.getLogit2Weights());
                double[] x0_Means=Utils.convertDoubles(triplet0.getMeans());
                double[] x0_logPrecs=Utils.convertDoubles(triplet0.getLogPrecisions());

                Distribution q_logit2W=new GaussianD(x0_logit2W,S2_logitW);
                Distribution q_Means=new GaussianD(x0_Means,S2_Means);
                Distribution q_logPrecs=new GaussianD(x0_logPrecs,S2_logPrecs);

                ArrayList<Double> temp1=new ArrayList(q_logit2W.Draw(1).get(0));
                ArrayList<Object> temp2=new ArrayList<>(Utils.Logit2ToLogit(temp1));


                ArrayList<Object> temp_Means=new ArrayList<>(M);

                for(int j=0; j<x0_Means.length; j++){

                    Distribution q_Means_j;

                    if(j==0)
                        q_Means_j=new tGaussianD(x0_Means[j],S2_Means[j][j],-1*Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
                    else{

                        double mu=Math.max(x0_Means[j], (double) temp_Means.get(j-1));

                        q_Means_j=new tGaussianD(mu,S2_Means[j][j],(double) temp_Means.get(j-1), Double.POSITIVE_INFINITY);
                    }

                    temp_Means.addAll(q_Means_j.Draw(1).get(0));
                }

                temp2.addAll(temp_Means);
                temp2.addAll(q_logPrecs.Draw(1).get(0));

                temp2.addAll(x0.get(i).subList(3*M-1,3*M));

                x.add(i,temp2);
            }
        }

        return x;
    }
}

class Proposal3GaussT extends Kernel{

    private double[][] S2_logitW;
    private double[][] S2_Means;
    private double[][] S2_logPrecs;

    Proposal3GaussT(double s2_logitW, double s2_Means, double s2_logPrecs, int d) {

        S2_logitW=Utils.idMatrix(s2_logitW/(3*d-1),d-1);
        S2_Means=Utils.idMatrix(s2_Means/(3*d-1),d);
        S2_logPrecs=Utils.idMatrix(s2_logPrecs/(3*d-1),d);
    }

    public void setPropVarFactor(double factor){

        this.S2_logitW=Utils.prod(this.S2_logitW,factor);
        this.S2_Means=Utils.prod(this.S2_Means,factor);
        this.S2_logPrecs=Utils.prod(this.S2_logPrecs,factor);
    }

    public double logDensity(List<Object> x, List<Object> x0) {

        double logDetJacobn=0, logDens=0;

        TripletMixGauss triplet=new TripletMixGauss(x.subList(0,x.size()-1));
        TripletMixGauss triplet0=new TripletMixGauss(x0.subList(0,x0.size()-1));

        double[] x0_Means=Utils.convertDoubles(triplet0.getMeans());
        double[] x0_logPrecs=Utils.convertDoubles(triplet0.getLogPrecisions());

        Distribution q_Means=new GaussianD(x0_Means,S2_Means);
        Distribution q_logPrecs=new GaussianD(x0_logPrecs,S2_logPrecs);

        List Means=triplet.getMeans();
        List logPrecisions=triplet.getLogPrecisions();

        int M=x.size()/3;

        if(M==1){

            logDetJacobn=-1*((double) logPrecisions.get(M-1));
            logDens=q_Means.logDensity(Means)+q_logPrecs.logDensity(logPrecisions);
        }
        else{

            double[] x0_logitW=Utils.convertDoubles(triplet0.getLogitWeights());

            List logitWeights=triplet.getLogitWeights();
            List logWeights=triplet.getLogWeightsFull();

            for(int i=0; i<x0_Means.length; i++){

                if(i<x0_logitW.length){

                    Distribution q_logitW_i;

                    if(i==0)
                        q_logitW_i=new tGaussianD(x0_logitW[i],S2_logitW[i][i],-1*Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
                    else{

                        double ub=Utils.LogToLogit(Math.log(1-Math.exp(Utils.logSumExp(logWeights.subList(0,i)))));
                        double mu=Math.min(x0_logitW[i],ub);

                        q_logitW_i=new tGaussianD(mu,S2_logitW[i][i],-1*Double.POSITIVE_INFINITY, ub);
                    }

                    logDens+=q_logitW_i.logDensity(logitWeights.subList(i,i+1));
                }

                Distribution q_Means_i;

                if(i==0)
                    q_Means_i=new tGaussianD(x0_Means[i],S2_Means[i][i],-1*Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
                else{

                    double mu=Math.max(x0_Means[i],(double) Means.get(i-1));

                    q_Means_i=new tGaussianD(mu,S2_Means[i][i],(double) Means.get(i-1), Double.POSITIVE_INFINITY);
                }

                logDens+=q_Means_i.logDensity(Means.subList(i,i+1));
            }

            logDens+=q_logPrecs.logDensity(logPrecisions);

            for(int i=0; i<M-1; i++) {

                logDetJacobn+=-1*((double) logWeights.get(i)
                        +Math.log(1.0-Math.exp((double) logWeights.get(i)))
                        +(double) logPrecisions.get(i));
            }
            logDetJacobn+=-1*((double) logPrecisions.get(M-1));
        }

        return logDens+logDetJacobn;
    }

    public List<ArrayList<Object>> Draw(List<ArrayList<Object>> x0) {

        List<ArrayList<Object>> x=new ArrayList(x0.size());

        for(int i=0; i<x0.size(); i++){

            int M=(x0.get(i).size())/3;

            if(M==1){

                double[] x0_Means=Utils.convertDoubles(x0.get(i).subList(0,1));
                double[] x0_logPrecs=Utils.convertDoubles(x0.get(i).subList(1,2));

                Distribution q_Means=new GaussianD(x0_Means,S2_Means);
                Distribution q_logPrecs=new GaussianD(x0_logPrecs,S2_logPrecs);

                ArrayList temp=new ArrayList<>(q_Means.Draw(1).get(0));
                temp.addAll(q_logPrecs.Draw(1).get(0));

                temp.addAll(x0.get(i).subList(2,3));

                x.add(i,temp);
            }
            else{

                double[] x0_logitW=Utils.convertDoubles(x0.get(i).subList(0,M-1));
                double[] x0_Means=Utils.convertDoubles(x0.get(i).subList(M-1,2*M-1));
                double[] x0_logPrecs=Utils.convertDoubles(x0.get(i).subList(2*M-1,3*M-1));

                Distribution q_logPrecs=new GaussianD(x0_logPrecs,S2_logPrecs);

                ArrayList<Object> temp=new ArrayList<>(3*M-1), temp_logitW=new ArrayList<>(M-1), temp_Means=new ArrayList<>(M);

                for(int j=0; j<x0_Means.length; j++){

                    if(j<x0_logitW.length){

                        Distribution q_logitW_j;

                        if(j==0)
                            q_logitW_j=new tGaussianD(x0_logitW[j],S2_logitW[j][j],-1*Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
                        else{

                            List prev=new ArrayList(temp_logitW.subList(0,j));
                            prev=Utils.LogitToLog(prev);
                            double ub=Utils.LogToLogit(Math.log(1-Math.exp(Utils.logSumExp( prev ))));
                            //double ub=Double.POSITIVE_INFINITY;

                            double mu=Math.min(ub,x0_logitW[j]);

                            q_logitW_j=new tGaussianD(mu,S2_logitW[j][j],-1*Double.POSITIVE_INFINITY, ub);
                        }

                        temp_logitW.addAll(q_logitW_j.Draw(1).get(0));
                    }

                    Distribution q_Means_j;

                    if(j==0)
                        q_Means_j=new tGaussianD(x0_Means[j],S2_Means[j][j],-1*Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
                    else{

                        double mu=Math.max(x0_Means[j], (double) temp_Means.get(j-1));

                        q_Means_j=new tGaussianD(mu,S2_Means[j][j],(double) temp_Means.get(j-1), Double.POSITIVE_INFINITY);
                        //q_Means_j=new tGaussianD(x0_Means[j],S2_Means[j][j],-1*Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
                    }

                    temp_Means.addAll(q_Means_j.Draw(1).get(0));
                }

                temp.addAll(temp_logitW);
                temp.addAll(temp_Means);
                temp.addAll(q_logPrecs.Draw(1).get(0));

                temp.addAll(x0.get(i).subList(3*M-1,3*M));

                x.add(i,temp);
            }
        }

        return x;
    }
}