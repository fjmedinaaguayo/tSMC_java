import java.util.*;

public class ParticleSet {

    public List<Double> logW;
    public List<theta> th;

    ParticleSet(){

        List<Double> logW_= new ArrayList();
        List<theta> th_= new ArrayList();

        this.logW=logW_;
        this.th=th_;
    }

    ParticleSet(int N){

        List<Double> logW_= new ArrayList(N);
        List<theta> th_= new ArrayList(N);

        this.logW=logW_;
        this.th=th_;
    }

    ParticleSet(List<Double> logW_, List<ArrayList<Object>> x_){

        int N=logW_.size();

        List<theta> th_= new ArrayList(N);

        for(int i=0; i<N; i++){
            th_.add(i,new theta(x_.get(i)));
        }

        this.logW=logW_;
        this.th=th_;
    }

    void setU(List<ArrayList<Object>> u){

        int N=this.th.size();

        if(u.size()>0){

            for(int i=0; i<N; i++)
                this.th.get(i).setU(u.get(i));
        }
    }

    public List<Double> normalisedWeights() {

        int N = this.logW.size();
        double sum= Utils.logSumExp(logW);
        List<Double> logWnorm=new ArrayList(N);

        for(int i=0; i<N; i++){
            logWnorm.add(i, logW.get(i)-sum);
        }

        return logWnorm;
    }

    public double ESS(){

        List<Double> logWnorm=this.normalisedWeights();
        double ess=0;
        for(int i=0; i<logWnorm.size(); i++){

            ess+=Math.exp(2*logWnorm.get(i));
        }

        return 1.0/ess;
    }

    public double relCESS(List<Double> logWinc){

        double logCESS=this.logRelCESS(logWinc);

        return Math.exp(logCESS);
    }

    public double logRelCESS(List<Double> logWinc){

        List<Double> logWnorm=this.normalisedWeights();

        return -1*Utils.logSumExp(Utils.sum(
                Utils.sum(logWnorm,Utils.prod(logWinc,2)), - 2*Utils.logSumExp(Utils.sum(logWnorm,logWinc))));
    }

    public double logRelCESS_mod(List<Double> logWinc){

        double logCESS=this.logRelCESS(logWinc);

        return Math.log(Math.abs(logCESS));
    }

    List<ArrayList<Object>> xList() {

        int N=this.th.size();

        List val=new ArrayList<ArrayList<Object>>();

        for(int i=0; i<N; i++)
            val.add(this.th.get(i).x);

        return val;

    }

    List<ArrayList<Object>> uList() {

        int N=this.th.size();

        List val=new ArrayList<ArrayList<Object>>();

        for(int i=0; i<N; i++)
            val.add(this.th.get(i).u);

        return val;

    }

    List<Double> estimates(){

        int N=this.th.size();
        int k=this.th.get(0).x.size();

        List<Double> val=new ArrayList<>(k);
        List<Double> logNWeights=this.normalisedWeights();

        for(int j=0;j<k; j++)
            val.add(0.0);

        for(int i=0; i<N; i++)
            for(int j=0; j<k; j++)
                val.set(j,val.get(j)+((double) this.th.get(i).x.get(j))*Math.exp(logNWeights.get(i)));

        return val;
    }

    List<Double> MixGaussEstimates(){

        int N=this.th.size();
        int k=this.th.get(0).x.size()/3;

        List<Double> val=new ArrayList<>(3*k);
        List<Double> logNWeights=this.normalisedWeights();

        for(int j=0;j<3*k; j++)
            val.add(0.0);

        for(int i=0; i<N; i++){

            TripletMixGauss triplet=new TripletMixGauss(th.get(i).x.subList(0,3*k-1));

            List weights=Utils.exp(triplet.getLogWeightsFull());
            List means=triplet.getMeans();
            List precisions=Utils.exp(triplet.getLogPrecisions());

            for(int j=0; j<k; j++){

                val.set(j,val.get(j)+((double) weights.get(j))*Math.exp(logNWeights.get(i)));
                val.set(k+j,val.get(k+j)+((double) means.get(j))*Math.exp(logNWeights.get(i)));
                val.set(2*k+j,val.get(2*k+j)+((double) precisions.get(j))*Math.exp(logNWeights.get(i)));
            }
        }

        return val;
    }

    double[][] SigmaMatrix(){

        int N=this.th.size();
        int k=this.th.get(0).x.size()/3;

        double[][] Sigma=new double[3*k-1][3*k-1];
        List<Double> logNWeights=this.normalisedWeights();
        List<Double> single_means=new ArrayList<>(3*k-1);

        for(int j=0;j<3*k-1; j++){
            single_means.add(0.0);

            for(int h=0; h<3*k-1; h++)
                Sigma[j][h]=0.0;
        }


        for(int i=0; i<N; i++){

            TripletMixGauss triplet=new TripletMixGauss(th.get(i).x.subList(0,3*k-1));
            List<Double> Particle_i= triplet.asSingleVectorLogit2();

            for(int j=0; j<3*k-1; j++){

                single_means.set(j,single_means.get(j)+Particle_i.get(j)*Math.exp(logNWeights.get(i)));
                for(int h=0; h<3*k-1; h++){
                    Sigma[j][h]+=Particle_i.get(j)*Particle_i.get(h)*Math.exp(logNWeights.get(i));
                }
            }
        }

        for(int j=0;j<3*k-1; j++){

            for(int h=0; h<3*k-1; h++)
                Sigma[j][h]-=single_means.get(j)*single_means.get(h);
        }

        return Sigma;
    }
}

class theta {

    List<Object> x;
    List<Object> u;
    int sizeX;
    int sizeU;

    theta(List<Object> x){

        this.x=new ArrayList(x);
        this.sizeX=x.size();
    }

    theta(List<Object> x, List<Object> u){

        this.x=new ArrayList(x);
        this.u=new ArrayList(u);
        this.sizeX=x.size();
        this.sizeU=u.size();
    }

    void setU(List<Object> u){

        this.u=u;
        this.sizeU=u.size();
    }

    List<Object> asList() {

        List val=new ArrayList<Object>();
        val.addAll(this.x);
        if(this.u!=null)
            val.addAll(this.u);

        return val;
    }
}