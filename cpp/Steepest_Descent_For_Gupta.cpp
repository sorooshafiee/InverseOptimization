using namespace std;

#include <string>
#include <vector>
#include "matrix.h"
#include "mex.h"
#include "math.h"

double * distSqrd(const mxArray *X1, double *X2)
{
    size_t n = mxGetM(X1);
    size_t N1 = mxGetN(X1);
    double *D2 = (double *) mxMalloc(N1 * sizeof(double));
    for (int i = 0; i < N1; i++){
        double acc = 0.0;
        for (int k = 0; k < n; k++){
            double tmp1 = mxGetPr(X1)[k+i*n];
            double tmp2 = X2[k];
            acc += pow(tmp1-tmp2,2);
        }
        D2[i] = acc;
    }
    return D2;
}

double * Kernel_Function(const mxArray *X1, double *X2, char *kernel_, double lengthScale)
{
    size_t n = mxGetM(X1);
    size_t N1 = mxGetN(X1);
    double eta = 1/pow(lengthScale,2);
    double *K = (double *) mxMalloc(N1 * sizeof(double));
    if (strcmp(kernel_,"linear") == 0){
        for (int i = 0; i < N1; i++){
            double acc = 0.0;
            for (int k = 0; k < n; k++){
                double tmp1 = mxGetPr(X1)[k+i*n];
                double tmp2 = X2[k];
                acc += tmp1*tmp2;
            }
            K[i] = acc;
        }
    }
    else if (strncmp(kernel_,"poly",4) == 0){
        char *temp = kernel_ + 4;
        int p;
        p = stoi(temp);
        for (int i = 0; i < N1; i++){
            double acc = 0.0;
            for (int k = 0; k < n; k++){
                double tmp1 = mxGetPr(X1)[k+i*n];
                double tmp2 = X2[k];
                acc += tmp1*tmp2;
            }
            K[i] = pow(acc+1,p);
        }
    }
    else if(strcmp(kernel_,"gauss") == 0){
        double *D2 = distSqrd(X1, X2);
        for (int i = 0; i < N1; i++){
            double tmp = -eta*D2[i];
            K[i] = exp(tmp);
        } 
        mxFree(D2);
    }
    else
        mexErrMsgTxt ("Unrecognised kernel function type.");
    return K;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
#define param       prhs[0]
#define alpha_in    prhs[1]
#define x_tr_in     prhs[2]
#define s_in        prhs[3]
#define optimal     plhs[0]
    
    if (nrhs != 4) {
        mexErrMsgTxt("Four inputs arguments required.");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }

    mxArray *kernel_in  = mxGetField(param, 0, "kernel");
    mxArray *sigma_in   = mxGetField(param, 0, "sigma");
    
    size_t n   = mxGetM(alpha_in);
    size_t Ntr = mxGetN(alpha_in);
    size_t m   = mxGetM(s_in);
    size_t Nt  = mxGetN(s_in);          
    size_t kernel_len = mxGetN(kernel_in) + 1;

    /* Define Fixed Vectors */
    char   *kernel_ = (char*)   mxMalloc(kernel_len);
    double *alpha   = (double*) mxMalloc(n * Ntr * sizeof(double));    
    double *s       = (double*) mxMalloc(m * Nt * sizeof(double));

    double status = mxGetString(kernel_in, kernel_, (mwSize)kernel_len);  
    double sigma = mxGetScalar(sigma_in);   
    for (int i=0; i<Ntr; i++)
        for (int j=0; j<n; j++)
            alpha[j+i*n] = mxGetPr(alpha_in)[j+i*n];
    for (int i=0; i<Nt; i++)
        for (int j=0; j<m; j++)
            s[j+i*m] = mxGetPr(s_in)[j+i*m];

    /* Initialization */
    const char *field_name[] = {"x"};
    optimal = mxCreateStructMatrix(Nt, 1, 1, field_name);         
    
    for (int k=0; k<Nt; k++){
        mxArray *x_out = mxCreateDoubleMatrix(n, 1, mxREAL);        
        double *x = mxGetPr(x_out);
        for (int i=0; i<n; i++)
            x[i] = 5 * ((double) rand() / (RAND_MAX));
        for (int r=0; r<1e5; r++){
            double *K = Kernel_Function(x_tr_in,x,kernel_,sigma);
            for (int j=0; j<n; j++){
                double acc = 0.0;
                for (int i=0; i<Ntr; i++)
                    acc += alpha[j+i*n] * K[i];
                x[j] = x[j] - 1e-3 * (s[j+k*n] + acc);
                if (x[j] > 5)
                    x[j] =5;
                if (x[j] < 0)
                    x[j] = 0;
            }
            mxFree(K);
        }
        mxSetField(optimal, k, "x", x_out);
    }
    return;
}