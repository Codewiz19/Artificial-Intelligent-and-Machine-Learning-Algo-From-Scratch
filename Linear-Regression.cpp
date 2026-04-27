 #include <bits/stdc++.h>
using namespace std;


double DotProduct ( vector<double> A , vector<double> B){
    int n = A.size() ;

    double result = 0 ;

    for(int i = 0 ; i < n ; i++ ){
        result += A[i] * B[i] ;
    }

    return result ;
}



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


class MultipleRegression {
private:
    vector<double> weights;
    double b; // Changed to double
    double lr;
    int epoch;

public:
    MultipleRegression(double lr = 0.001, int epoch = 100) {
        this->lr = lr;
        this->epoch = epoch;
        this->b = 0.0; // Initialize bias
    }

    void fit(const vector<vector<double>>& x, const vector<double>& y) {
        int datasize = x.size();          // Number of rows (samples)
        if (datasize == 0) return;
        int features = x[0].size();       // Number of columns (features)

        weights.assign(features, 0.0);    // Initialize weights to 0.0

        for (int i = 0; i < epoch; i++) {
            
            // dw must be a vector because each feature needs its own gradient
            vector<double> dw(features, 0.0); 
            double db = 0.0;

            for (int j = 0; j < datasize; j++) {
                double y_predicted = predict(x[j]);
                double error = y_predicted - y[j]; // standard definition for gradient calculation
                
                for (int k = 0; k < features; k++) {
                    dw[k] += error * x[j][k];
                }
                db += error;
            }
            
            // Update weights and bias using the learning rate (Batch Gradient Descent)
            for (int k = 0; k < features; k++) {
                weights[k] -= lr * (dw[k] / datasize);
            }
            b -= lr * (db / datasize);

            if(epoch%100 == 0) cout<<LossMR(x,y)<<endl;
        }
    }

    double LossMR(const vector<vector<double>>& x, const vector<double>& y) {
        int datasize = y.size();
        double total_loss = 0.0;

        for (int i = 0; i < datasize; i++) {
            double y_predicted = predict(x[i]);
            double error = y[i] - y_predicted;
            total_loss += (error * error);
        }

        return total_loss / datasize;
    }

    void printBias() { 
        cout <<" Bias " <<  b << endl;
    }

    void printWeights() {
        int i = 1 ;
        for (double w : weights){
            cout << "Weights"<<i<<" : " << w << " ";
            i++;
        }

        cout << endl;
    }

    double predict(const vector<double>& features) {
        return DotProduct(weights, features) + b;
    }
};


 


int main(){

    // Example for Linear regressor 

    // vector<double> x = {1,2,3,4,5,6,7,8,9,10,11,12} ;
    // vector<double> y = {2,4,6,8,10,12,14,16,18,20,22,24} ;

    // LinearRegression lr(0.001 , 10000) ;
    // lr.fit(x,y) ;

    // lr.printweightandbias() ;
    // cout << "Predicted value for 6 : " << lr.predict(300) << endl ;



    // Example for Multiple regressor 

        vector<vector<double>> X = {
            {1, 2, 1},
            {2, 1, 2},
            {3, 4, 1},
            {4, 3, 6},
            {5, 5, 2},
            {5 ,6, 1} ,
            {6 ,7, 10} ,
        };

        vector<double> Y = {
            12,   // 2*1 + 3*2 + 4*1= 8
            15,   // 2*2 + 3*1 + 4*2 = 7
            22,
            41,
            33, 
            32,
            73,   
        };

        MultipleRegression MR(0.01, 10000) ;

        MR.fit(X,Y) ;

        MR.printBias();
        MR.printWeights();


        cout<<MR.predict({10,6}) ;



    return 0 ;
}