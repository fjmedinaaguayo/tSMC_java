import java.util.*;

interface Transform {

    List<Object> fn(theta th);
    theta fnInv(List<Object> y);
    List<ArrayList<Object>> fn(List<theta> th);
    List<theta> fnInvList(List<ArrayList<Object>> y);
    double logDetJacobnInv(List<Object> y);
}

class T1 implements Transform {

    private double x0;
    private double sx;
    private double u0;
    private double su;

    T1(double x0_, double u0_, double sx_, double su_){
        this.x0=x0_;
        this.sx=sx_;
        this.u0=u0_;
        this.su=su_;
    }

    public List<Object> fn(theta th){

        List val= new ArrayList<Double[]>();
        val.addAll(Utils.sum(Utils.prod(th.x,sx),x0));
        val.addAll(Utils.sum(Utils.prod(th.u,su),u0));

        return val;
    }

    public theta fnInv(List<Object> y){

        int xSize=(int) y.get(y.size()-1);

        List x=Utils.prod(Utils.sum(y.subList(0,xSize),-x0),1.0/sx);
        List u=Utils.prod(Utils.sum(y.subList(xSize,y.size()),-u0),1.0/su);

        theta th=new theta(x,u);

        return th;
    }

    public List<ArrayList<Object>> fn(List<theta> th){

        List val = new ArrayList<>(th.size());

        for(int i=0; i<th.size(); i++){
            val.add(this.fn(th.get(i)));
        }

        return val;
    }

    public List<theta> fnInvList(List<ArrayList<Object>> y){

        List val = new ArrayList<theta>(y.size());

        for(int i=0; i<y.size(); i++){
            val.add(this.fnInv(y.get(i)));
        }

        return val;
    }

    public double logDetJacobn_aux(theta th){

        int xDim=th.x.size(), uDim=th.u.size();
        int dim=xDim+uDim;
        double[][] J=Utils.idMatrix(1.0, dim);

        return xDim*Math.log(sx)+uDim*Math.log(su);
    }

    public double logDetJacobnInv(List<Object> y){

        theta th=fnInv(y);
        double logDetJ=this.logDetJacobn_aux(th);

        return -1*logDetJ;
    }
}

class BirthTransf implements Transform {

    BirthTransf(){

    }

    public List<Object> fn(theta th){

        int xSize=th.x.size();

        if(th.u.size()==1){

            TripletMixGauss triplet=new TripletMixGauss(th.x.subList(0,xSize));
            List y=triplet.asSingleVector();
            List labels=new ArrayList(Arrays.asList(th.u.get(0)));
            y.add(labels);

            return y;
        }

        TripletMixGauss triplet=new TripletMixGauss(th.x.subList(0,xSize-1));

        double new_logW=Utils.LogitToLog((double) th.u.get(1));

        int position=triplet.findPosition((double) th.u.get(2));

        triplet.rescaleExtend(new_logW,(double) th.u.get(2),(double) th.u.get(3),position);

        List y=triplet.asSingleVector();

        List labels=new ArrayList(Arrays.asList(th.u.get(0),position));
        y.add(labels);

//        this.fnInv(y);

        if(Utils.areOrdered(triplet.getMeans())!=1)
            System.out.println("Means not ordered!");

        return y;
    }

    public theta fnInv(List<Object> y){

        int ySize=y.size();
        List labels=(List) y.get(ySize-1);

        if(labels.size()==1){

            theta th=new theta(y.subList(0,ySize-1),labels);
            return th;
        }

        TripletMixGauss triplet=new TripletMixGauss(y.subList(0,ySize-1));
        if(triplet.areWeightsValid()==0){

            return null;
        }

        int position=(int) labels.get(1);

        List u=new ArrayList(Arrays.asList(labels.get(0)));
        u.addAll(triplet.subTriplet_logitW(position));
        triplet.rescaleDelete(position);
        List x=triplet.asSingleVector();
        x.add("NA");

        theta th=new theta(x,u);

        return th;
    }

    public List<ArrayList<Object>> fn(List<theta> th){

        List val = new ArrayList(th.size());

        for(int i=0; i<th.size(); i++){
            val.add(this.fn(th.get(i)));
        }

        return val;
    }

    public List<theta> fnInvList(List<ArrayList<Object>> y){

        List val = new ArrayList<theta>(y.size());

        for(int i=0; i<y.size(); i++){
            val.add(this.fnInv(y.get(i)));
        }

        return val;
    }

    public double logDetJacobn_aux(theta th){

        int M=th.x.size()/3;

        if(th.u.size()==1)
            return 0.0;

        double new_logitW=(double) th.u.get(1);
        double new_logW=-1*Math.log(1+Math.exp(-1*new_logitW));

        return (M-1)*Math.log(1.0-Math.exp(new_logW));
    }

    public double logDetJacobnInv(List<Object> y){

        int ySize=y.size();
        List labels=(List) y.get(ySize-1);

        if(labels.size()==1)
            return 0.0;

        theta th=fnInv(y);
        double logDetJ=this.logDetJacobn_aux(th);

        return -logDetJ+Math.log(ySize/3);
    }
}

class SplitTransf implements Transform {

    SplitTransf(){

    }

    public List<Object> fn(theta th){

        int xSize=th.x.size();

        if(th.u.size()==1){

            TripletMixGauss triplet=new TripletMixGauss(th.x.subList(0,xSize));
            List y=triplet.asSingleVector();
            List labels=new ArrayList(Arrays.asList(th.u.get(0)));
            y.add(labels);

            return y;
        }

        TripletMixGauss triplet=new TripletMixGauss(th.x.subList(0,xSize-1));

        double u1=(double) th.u.get(1), u2=(double) th.u.get(2), u3=(double) th.u.get(3);
        int k=(int) th.u.get(4);

        List logWeights=triplet.getLogWeightsFull();

        double k_logW=(double) logWeights.get(k-1),
                k_mean=triplet.getMeans().get(k-1),
                k_logPrec=triplet.getLogPrecisions().get(k-1);

        double km_logW=Math.log(u1)+k_logW, kp_logW=Math.log(1-u1)+k_logW;
        double km_mean=k_mean-u2*Math.exp(-0.5*k_logPrec)*Math.sqrt(Math.exp(kp_logW-km_logW)),
                kp_mean=k_mean+u2*Math.exp(-0.5*k_logPrec)*Math.sqrt(Math.exp(km_logW-kp_logW));
        double km_logPrec=-1*(Math.log(u3)+Math.log(1-Math.pow(u2,2))-k_logPrec+k_logW-km_logW),
                kp_logPrec=-1*(Math.log(1-u3)+Math.log(1-Math.pow(u2,2))-k_logPrec+k_logW-kp_logW);

        triplet.delete(k-1);

        int position1=triplet.findPosition(km_mean);
        triplet.extend(km_logW,km_mean,km_logPrec,position1);

        int position2=triplet.findPosition(kp_mean);
        triplet.extend(kp_logW,kp_mean,kp_logPrec,position2);

        List y=triplet.asSingleVector();

        List positions=new ArrayList<Integer>(2);
        positions.add(position1);
        positions.add(position2);

        List labels=new ArrayList(Arrays.asList(th.u.get(0),positions));
        y.add(labels);

//        theta check=this.fnInv(y);

        if(Utils.areOrdered(triplet.getMeans())!=1)
            System.out.println("Means not ordered!");

        return y;
    }

    public theta fnInv(List<Object> y){

        int ySize=y.size();
        List labels=(List) y.get(ySize-1);

        if(labels.size()==1){

            theta th=new theta(y.subList(0,ySize-1),labels);
            return th;
        }

        TripletMixGauss triplet=new TripletMixGauss(y.subList(0,ySize-1));
        if(triplet.areWeightsValid()==0){

            return null;
        }

        List positions=(List) labels.get(1);
        int position1=(int) positions.get(0);
        int position2=(int) positions.get(1);

        List yp=triplet.subTriplet_logW(position2);
        triplet.delete(position2);

        List ym=triplet.subTriplet_logW(position1);
        triplet.delete(position1);

        double km_logW=(double) ym.get(0), kp_logW=(double) yp.get(0);
        double km_mean=(double) ym.get(1), kp_mean=(double) yp.get(1);
        double km_logPrec=(double) ym.get(2), kp_logPrec=(double) yp.get(2);

        double k_logW=Math.log(Math.exp(km_logW)+Math.exp(kp_logW)),
                k_mean=(Math.exp(km_logW)*km_mean+Math.exp(kp_logW)*kp_mean)*Math.exp(-k_logW),
                k_logPrec=-1*Math.log( ( Math.exp(km_logW)*(Math.pow(km_mean,2)+Math.exp(-km_logPrec))
                        +Math.exp(kp_logW)*(Math.pow(kp_mean,2)+Math.exp(-kp_logPrec)) )*Math.exp(-k_logW)
                        -Math.pow(k_mean,2) );

        int k=triplet.findPosition(k_mean);
        triplet.extend(k_logW,k_mean,k_logPrec,k);

        List x=triplet.asSingleVector();
        x.add("NA");

        double u1=Math.exp(km_logW-k_logW),
                u2=(k_mean-km_mean)*Math.exp(0.5*k_logPrec)*Math.sqrt(u1/(1-u1)),
                u3=u1/(1-Math.pow(u2,2))*Math.exp(k_logPrec-km_logPrec);

        List u=new ArrayList<>(5);

        u.add(labels.get(0));
        u.add(u1);
        u.add(u2);
        u.add(u3);
        u.add(k+1);

        theta th=new theta(x,u);

//        List check=this.fn(th);

        return th;
    }

    public List<ArrayList<Object>> fn(List<theta> th){

        List val = new ArrayList<Object>(th.size());

        for(int i=0; i<th.size(); i++){
            val.add(this.fn(th.get(i)));
        }

        return val;
    }

    public List<theta> fnInvList(List<ArrayList<Object>> y){

        List val = new ArrayList<theta>(y.size());

        for(int i=0; i<y.size(); i++){
            val.add(this.fnInv(y.get(i)));
        }

        return val;
    }

    public double logDetJacobn_aux(theta th){

        if(th.u.size()==1)
            return 0.0;

        int xSize=th.x.size();

        TripletMixGauss triplet=new TripletMixGauss(th.x.subList(0,xSize-1));

        double u1=(double) th.u.get(1), u2=(double) th.u.get(2), u3=(double) th.u.get(3);
        int k=(int) th.u.get(4);

        List logWeights= triplet.getLogWeightsFull();

        double k_logW=(double) logWeights.get(k-1),
                k_mean=triplet.getMeans().get(k-1),
                k_logPrec=triplet.getLogPrecisions().get(k-1);

        double km_logW=Math.log(u1)+k_logW, kp_logW=Math.log(1-u1)+k_logW;
        double km_mean=k_mean-u2*Math.exp(-0.5*k_logPrec)*Math.sqrt(Math.exp(kp_logW-km_logW)),
                kp_mean=k_mean+u2*Math.exp(-0.5*k_logPrec)*Math.sqrt(Math.exp(km_logW-kp_logW));
        double km_logPrec=-1*(Math.log(u3)+Math.log(1-Math.pow(u2,2))-k_logPrec+k_logW-km_logW),
                kp_logPrec=-1*(Math.log(1-u3)+Math.log(1-Math.pow(u2,2))-k_logPrec+k_logW-kp_logW);


        //The Jacobian has been modified in order to account the transformation to precisions from variances.
        double value=(km_logPrec+kp_logPrec-k_logPrec)+Math.log(Math.exp(km_logW)+Math.exp(kp_logW))
                +Math.log(kp_mean-km_mean)-Math.log(u2)-Math.log(u3)-Math.log(1-Math.pow(u2,2))
                -Math.log(1-u3);

        return value;
    }

    public double logDetJacobnInv(List<Object> y){

        int ySize=y.size();
        List labels=(List) y.get(ySize-1);

        if(labels.size()==1)
            return 0.0;

        theta th=fnInv(y);
        double logDetJ=this.logDetJacobn_aux(th);

        return -logDetJ+Math.log(ySize/3*(ySize/3-1)/2);
    }

}

class MixTransf implements Transform {

    BirthTransf Birth;
    SplitTransf Split;

    MixTransf(){

        this.Birth=new BirthTransf();
        this.Split=new SplitTransf();
    }

    public List<Object> fn(theta th){

        if((int) th.u.get(0)==1){

            return Birth.fn(th);
        }
        else{

            return Split.fn(th);
        }
    }

    public theta fnInv(List<Object> y){

        int ySize=y.size();
        List labels=(List) y.get(ySize-1);

        if((int) labels.get(0)==1){

            return Birth.fnInv(y);
        }
        else{

            return Split.fnInv(y);
        }
    }

    public List<ArrayList<Object>> fn(List<theta> th){

        List val = new ArrayList(th.size());

        for(int i=0; i<th.size(); i++){
            val.add(this.fn(th.get(i)));
        }

        return val;
    }

    public List<theta> fnInvList(List<ArrayList<Object>> y){

        List val = new ArrayList<theta>(y.size());

        for(int i=0; i<y.size(); i++){
            val.add(this.fnInv(y.get(i)));
        }

        return val;
    }

    public double logDetJacobn_aux(theta th){

        if((int) th.u.get(0)==1){

            return Birth.logDetJacobn_aux(th);
        }
        else{

            return Split.logDetJacobn_aux(th);
        }
    }

    public double logDetJacobnInv(List<Object> y){

        int ySize=y.size();
        List labels=(List) y.get(ySize-1);

        if((int) labels.get(0)==1){

            return Birth.logDetJacobnInv(y);
        }
        else{

            return Split.logDetJacobnInv(y);
        }
    }
}