import com.sun.jmx.remote.internal.ArrayQueue;
import org.apache.commons.math3.stat.descriptive.moment.Mean;

import java.util.*;

abstract class Problem {

    int T, t0;
    Distribution pi0;
    List<TargetDistn> pi;
    List<Distribution> psi;
    List<Transform> G;
    List<Kernel> K;

    double logPhi_t(int t, List<Object> y){

        if(t==0)
            return pi0.logDensity(y);
        else
            return pi.get(t-1).logDensity(y);
    }

//    double logPhi_tm1_t(int t, List<Object> y){
//
//        theta th=G.get(t-1).fnInv(y);
//
//        if(th==null)
//            return -1*Double.POSITIVE_INFINITY;
//        else if(t==1){
//
//            double temp=pi0.logDensity(th.x)+psi.get(0).logDensity(th.u);
//            if(temp!=-1*Double.POSITIVE_INFINITY)
//                temp+=G.get(0).logDetJacobnInv(y);
//
//            return temp;
//        }
//        else{
//
//            double temp=pi.get(t-2).logDensity(th.x)+psi.get(t-1).logDensity(th.u);
//            if(temp!=-1*Double.POSITIVE_INFINITY)
//                temp+=G.get(t-1).logDetJacobnInv(y);
//
//            return temp;
//        }
//    }

    double logPhi_tm1_t(int t, List<Object> y){

        theta th=G.get(t-1).fnInv(y);

        if(th==null)
            return -1*Double.POSITIVE_INFINITY;
        else if(t==1){

            double temp=pi0.logDensity(th.x)+psi.get(0).logDensity(th.u);
            if(temp!=-1*Double.POSITIVE_INFINITY)
                temp+=G.get(0).logDetJacobnInv(y);

            return temp;
        }
        else if(( pi.get(t-2).logDensity(th.x)+psi.get(t-1).logDensity(th.u) )!=-1*Double.POSITIVE_INFINITY ){

            List<Object> y_copy=G.get(t-1).fn(th);

            int ySize=y_copy.size();
            ArrayList<Object> aux=(ArrayList) y_copy.get(ySize-1);
            ArrayList<Double> temp=new ArrayList<>(ySize/3);



            if((int) aux.get(0)==0){

                ArrayList<Integer> labels=(ArrayList<Integer>) aux.get(1);
                int l1=labels.get(0), l2=labels.get(1);

                for(int i=0; i<ySize/3-1; i++){

                    for(int j=i+1; j<ySize/3; j++){

                        labels.set(0,i);
                        labels.set(1,j);

                        th=G.get(t-1).fnInv(y_copy);

                        double temp2=pi.get(t-2).logDensity(th.x)+psi.get(t-1).logDensity(th.u);
                        if(temp2==-1*Double.POSITIVE_INFINITY)
                            temp.add(temp2);
                        else
                            temp.add( temp2+G.get(t-1).logDetJacobnInv(y_copy));
                    }
                }

                labels.set(0,l1);
                labels.set(1,l2);
            }
            else{

                int l = (int) aux.get(1);

                for(int i=0; i<ySize/3; i++){

                    aux.set(1,i);
                    th=G.get(t-1).fnInv(y_copy);

                    double temp2=pi.get(t-2).logDensity(th.x)+psi.get(t-1).logDensity(th.u);
                    if(temp2==-1*Double.POSITIVE_INFINITY)
                        temp.add(temp2);
                    else
                        temp.add( temp2+G.get(t-1).logDetJacobnInv(y_copy));
                }

                aux.set(1,l);
            }

            return Utils.logSumExp(temp);
        }

        return -1*Double.POSITIVE_INFINITY;
    }

    double GeomlogPhi_t(double gamma, int t, List<Object> y){

        if(gamma==0){
            return this.logPhi_tm1_t(t, y);
        }
        else if(gamma==1){
            return this.logPhi_t(t, y);
        }

        return gamma*this.logPhi_t(t, y)+(1-gamma)*this.logPhi_tm1_t(t, y);
    }
}

class ToyExample extends Problem {

    ToyExample(int T_){

        this.T=T_;
        double s2=1.0;
        double[] mu={0.0};
        double[][] S2={{s2}};

        this.pi0=new GaussianD(mu,S2);
        List<TargetDistn> pi_= new ArrayList(T);
        List<Distribution> psi_= new ArrayList(T+1);
        List<Transform> G_= new ArrayList<>(T);
        List<Kernel> K_= new ArrayList(T);

        for(int t=1; t<=T; t++){

            TargetDistn d1=new GaussianD(Utils.equalVector(t,t*(t+1)/2 +1),Utils.idMatrix((t+1)*s2,t*(t+1)/2+1));
            pi_.add(d1);

            Distribution d2=new GaussianD(Utils.equalVector(0,t),Utils.idMatrix(4*s2,t));
            psi_.add(d2);

            Transform T=new T1(t-(t-1)*Math.sqrt((t+1.0)/t),t,
                    Math.sqrt((t+1.0)/t),Math.sqrt((t+1.0)/4));
            G_.add(T);

            Kernel q=new GaussianK(Utils.idMatrix(2*s2,t*(t+1)/2+1));
            Kernel Kt=new MCMCK(d1,q);
            K_.add(Kt);
        }

        this.pi=pi_;
        this.psi=psi_;
        this.G=G_;
        this.K=K_;
    }
}

class MixGaussian extends Problem {

    MixGaussian(int t0, int T, List<Double> Obs, double probBirth){

        this.t0=t0;
        this.T=T;
        int nObs=Obs.size();
        double s2_logitW=2.0, s2_Means=3.0, s2_logPrecs=2.0;

        double prior_mean=Utils.mean(Obs), prior_sd=Utils.range(Obs);
        double prior_shape=2, prior_rate=prior_shape*Math.pow(Utils.range(Obs)/10.0,2);
        List prior_params=new ArrayList<>(Arrays.asList(prior_mean,prior_sd,prior_shape,prior_rate));

        double alpha1=2, beta1=2, alpha2=2, beta2=2, alpha3=1, beta3=1;
        List split_params=new ArrayList<>(Arrays.asList(alpha1,beta1,alpha2,beta2,alpha3,beta3));

        this.pi0= new GaussGammaPrior(1,prior_mean,prior_sd,prior_shape,prior_rate);

        List<TargetDistn> pi_= new ArrayList(T);
        List<Distribution> psi_= new ArrayList(T);
        List<Transform> G_= new ArrayList<>(T);
        List<Kernel> K_= new ArrayList(T);

        for(int t=1; t<=T; t++){

            TargetDistn d1=new GaussMixPosterior(t, Obs, prior_mean, prior_sd,prior_shape, prior_rate);
            pi_.add(d1);

            List params=new ArrayList(Arrays.asList(prior_params,split_params));
            Distribution d2=new SMCProposal(t-1,probBirth,params);
            Transform Transf=new MixTransf();

            psi_.add(d2);
            G_.add(Transf);

            Kernel Kt=new Proposal3GaussLogit2(s2_logitW/nObs,
                    s2_Means/nObs,s2_logPrecs/nObs,t);
            K_.add(Kt);
        }

        this.pi=pi_;
        this.psi=psi_;
        this.G=G_;
        this.K=K_;
    }
}

class SMC2 extends Problem {

    SMC2(int K, List<Double> Obs){

        this.t0=0;
        this.T=1;
        int nObs=Obs.size();
        //double s2_logitW=2.0, s2_Means=3.0, s2_logPrecs=2.0;
        double s2_logitW=100.0, s2_Means=150.0, s2_logPrecs=100.0;

        double prior_mean=Utils.mean(Obs), prior_sd=Utils.range(Obs);
        double prior_shape=2, prior_rate=prior_shape*Math.pow(Utils.range(Obs)/10.0,2);
        List prior_params=new ArrayList<>(Arrays.asList(prior_mean,prior_sd,prior_shape,prior_rate));

        double alpha1=2, beta1=2, alpha2=2, beta2=2, alpha3=1, beta3=1;
        List split_params=new ArrayList<>(Arrays.asList(alpha1,beta1,alpha2,beta2,alpha3,beta3));

        this.pi0= new GaussGammaPrior(K,prior_mean,prior_sd,prior_shape,prior_rate);

        List<TargetDistn> pi_= new ArrayList(T);
        List<Distribution> psi_= new ArrayList(T);
        List<Transform> G_= new ArrayList<>(T);
        List<Kernel> K_= new ArrayList(T);

        for(int t=t0+1; t<=T; t++){

            TargetDistn d1=new GaussMixPosterior(K, Obs, prior_mean, prior_sd,prior_shape, prior_rate);
            pi_.add(d1);

            List params=new ArrayList(Arrays.asList(prior_params,split_params));
            Distribution d2=new SMCProposal(t-1,1,params);

            Transform T=new MixTransf();

            psi_.add(d2);
            G_.add(T);

            Kernel Kt=new Proposal3GaussLogit2(s2_logitW/nObs,
                    s2_Means/nObs,s2_logPrecs/nObs,K);
            K_.add(Kt);
        }

        this.pi=pi_;
        this.psi=psi_;
        this.G=G_;
        this.K=K_;
    }
}

class TripletMixGauss {

    private int size;
    private List<Double> logWeightsFull;
    private List<Double> Means;
    private List<Double> logPrecisions;

    TripletMixGauss(List x){

        int size=(x.size()+1)/3;
        this.size=size;

        List logWeights=new ArrayList<Double>(size);

        if(this.size==1){

            logWeights.add(0.0);
        }
        else{

            logWeights=Utils.LogitToLog(x.subList(0,size-1));
            logWeights.add(Math.log(1.0-Math.exp(Utils.logSumExp(logWeights))));

        }

        this.logWeightsFull=logWeights;
        this.Means=new ArrayList(x.subList(size-1,2*size-1));
        this.logPrecisions=new ArrayList(x.subList(2*size-1,3*size-1));
    }

    TripletMixGauss(List x,String type){

        int size=(x.size()+1)/3;
        this.size=size;

        List logWeights=new ArrayList<Double>(size);

        if(this.size==1){

            logWeights.add(0.0);
        }
        else{

            if(type=="Logit2"){

                logWeights=Utils.LogitToLog(Utils.Logit2ToLogit(x.subList(0,size-1)));
            }else{

                logWeights=Utils.LogitToLog(x.subList(0,size-1));
            }
            logWeights.add(Math.log(1.0-Math.exp(Utils.logSumExp(logWeights))));

        }

        this.logWeightsFull=logWeights;
        this.Means=new ArrayList(x.subList(size-1,2*size-1));
        this.logPrecisions=new ArrayList(x.subList(2*size-1,3*size-1));
    }

    TripletMixGauss(List<Double> logWeightsFull, List<Double> Means, List<Double> logPrecisions){

        this.size=logWeightsFull.size();
        this.logWeightsFull=logWeightsFull;
        this.Means=Means;
        this.logPrecisions=logPrecisions;
    }

    List<Double> getLogWeightsFull(){

        return this.logWeightsFull;
    }

    List<Double> getLogitWeights(){

        return Utils.LogToLogit(this.logWeightsFull.subList(0,this.size-1));
    }

    List<Double> getLogit2Weights(){

        List<Double> logW=new ArrayList(this.logWeightsFull);

        return Utils.sum(logW,-1*logW.get(this.size-1)).subList(0,this.size-1);
    }

    List<Double> getMeans(){

        return this.Means;
    }

    List<Double> getLogPrecisions(){

        return this.logPrecisions;
    }

    void delete(int i){

        this.logWeightsFull.remove(i);
        this.Means.remove(i);
        this.logPrecisions.remove(i);

        this.size-=1;
    }

    void extend(double new_logWeight, double new_Mean, double new_logPrecision, int i){

        this.logWeightsFull.add(i,new_logWeight);
        this.Means.add(i,new_Mean);
        this.logPrecisions.add(i,new_logPrecision);

        this.size+=1;
    }

    void rescaleExtend(double new_logWeight, double new_Mean, double new_logPrecision, int i){

        for(int j=0; j<this.size; j++)
            this.logWeightsFull.set(j,this.logWeightsFull.get(j)+Math.log(1-Math.exp(new_logWeight)) );

        this.extend(new_logWeight,new_Mean,new_logPrecision,i);
    }

    void rescaleDelete(int i){

        double logWeight_i=this.logWeightsFull.get(i);

        for(int j=0; j<this.size; j++)
            this.logWeightsFull.set(j,this.logWeightsFull.get(j)-Math.log(1-Math.exp(logWeight_i)) );

        this.delete(i);
    }

    int findPosition(double new_Mean){

        for(int i=0; i<this.size; i++){

            if(new_Mean<this.Means.get(i))
                return i;
        }

        return this.size;
    }

    List asSingleVector(){

        List y=new ArrayList(Utils.LogToLogit(this.logWeightsFull.subList(0,this.size-1)));

        y.addAll(this.Means);
        y.addAll(this.logPrecisions);

        return y;
    }

    List asSingleVectorLogit2(){

        List y=new ArrayList(this.getLogit2Weights());

        y.addAll(this.Means);
        y.addAll(this.logPrecisions);

        return y;
    }

    List subTriplet_logitW(int i){

        List u=new ArrayList<Double>(3);

        u.add(Utils.LogToLogit(this.logWeightsFull.get(i)));
        u.add(this.Means.get(i));
        u.add(this.logPrecisions.get(i));

        return u;
    }

    List subTriplet_logW(int i){

        List u=new ArrayList<Double>(3);

        u.add(this.logWeightsFull.get(i));
        u.add(this.Means.get(i));
        u.add(this.logPrecisions.get(i));

        return u;
    }

    int areWeightsValid(){

        if(Math.abs(Utils.logSumExp(this.logWeightsFull))>1e-9)
            return 0;

        for(int i=0; i<this.size; i++){

            if(this.logWeightsFull.get(i)>0 || Double.isNaN(this.logWeightsFull.get(i)))
                return 0;
        }

        return 1;
    }
}