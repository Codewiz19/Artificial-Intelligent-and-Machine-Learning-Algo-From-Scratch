 #include <bits/stdc++.h>
using namespace std;


class LinearRegression{
    private:
    double w , b  ;  // weight and bias
    float lr ;   // Learning Rate
    int epochs ;


    public :

    LinearRegression( float lr = 0.001 , int epochs = 100){
        this->w = 0.0 ;
        this->b = 0.0 ;
        this->lr = lr ;
        this->epochs = epochs ;
    }

    void printweightandbias(){
        cout << "Weight : " << w << " Bias : " << b << endl ;
    }

    void fit(vector<double> x , vector<double> y ){

        if(x.size() != y.size()){
            cout << "Size of x and y should be same " << endl ;
            return ;
        }

        int n = x.size() ;

        for(int i = 0 ; i<epochs ; i++){

            double dw = 0.0 , db =0.0 ;

            for(int j = 0 ; j<n ; j++){
                double y_predicted = w*x[j] + b ;

                dw += (y_predicted - y[j]) * x[j] ; // Gradient for weight 
                db += (y_predicted - y[j]) ; // Gradient for bias

                // both are derived from the MSE (Mean Squared Error) loss function of linear regression 
            }

            w = w - lr*dw/n ;
            b = b - lr*db/n ;

            if(i%1000 == 0){
                cout << "Epoch " << i << " Loss : " << loss(x,y) << endl ;
            }
        }

        
    }

    double predict(double x){
        return w*x + b ;
    }

    float loss(vector<double> x , vector<double> y){
        int n = x.size() ;
        float total_loss = 0.0 ;

        for(int i = 0 ; i<n ; i++){
            double y_predicted = w*x[i] + b ;
            total_loss += pow(y_predicted - y[i] , 2) ;
        }

        return total_loss/n ;
    }
    
    


} ;


int main(){
    vector<double> x = {1,2,3,4,5,6,7,8,9,10} ;
    vector<double> y = {2,4,6,8,10,12,14,16,18,20} ;

    LinearRegression lr(0.001 , 10000) ;
    lr.fit(x,y) ;

    lr.printweightandbias() ;
    cout << "Predicted value for 6 : " << lr.predict(300) << endl ;

    return 0 ;
}