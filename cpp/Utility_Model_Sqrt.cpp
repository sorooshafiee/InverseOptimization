#define ILOUSESTL
using namespace std;

#include <ilcplex/ilocplex.h>
#include <vector>
#include "mex.h"

ILOSTLBEGIN
typedef IloArray<IloNumArray>    NumMatrix;
typedef IloArray<IloNumVarArray> NumVarMatrix;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    #define param   prhs[0]
    #define A_in 	prhs[1]
	#define b_in 	prhs[2]
    #define s_in 	prhs[3]	
    
	#define optimal plhs[0]
    
    if (nrhs != 4) {
        mexErrMsgTxt("Four input arguments required.");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    mxArray *h_in       = mxGetField(param, 0, "h");
    mxArray *H_in       = mxGetField(param, 0, "H");
    mxArray *W_in       = mxGetField(param, 0, "W"); 

    size_t n	= mxGetNumberOfElements(b_in);
    size_t N 	= mxGetN(s_in);
    size_t m 	= mxGetM(s_in);
    size_t nh	= mxGetNumberOfElements(h_in);
    
	/* Initialization */
    const char *field_names[] = {"x","objective","diagnosis"};
    optimal = mxCreateStructMatrix(N, 1, 3, field_names); 
	
    IloEnv env;    
    try
    {
		/* Define Additional Fixed Vectors */        
		NumMatrix 	s(env,N);
		NumMatrix 	A(env,n);
		IloNumArray b(env,n);  
		NumMatrix 	W(env,nh);
        NumMatrix 	H(env,nh);
        IloNumArray h(env,nh);	  		
		for (int i=0; i < N; i++) {
			s[i] = IloNumArray(env,m);
			for (int j=0; j<m; j++)
                s[i][j] = mxGetPr(s_in)[j+i*m];
        }   
		for (int i=0; i<n; i++){
			A[i] = IloNumArray(env,n);
			b[i] = mxGetPr(b_in)[i];
			for (int j=0; j<n; j++)
				A[i][j] = mxGetPr(A_in)[i+j*n];
		}	
		for (int i=0; i < nh; i++) {
            H[i] 		= IloNumArray(env,m);
            W[i] 		= IloNumArray(env,n);
            h[i]        = mxGetPr(h_in)[i];
            for (int j=0; j<m; j++){
                H[i][j] = mxGetPr(H_in)[j*nh+i];
            }
            for (int j=0; j<n; j++){
                W[i][j] = mxGetPr(W_in)[j*nh+i];
            }
        }		
		
		for (int k=0; k<N; k++){		
			IloModel model(env);
			
			/* Define Decision Variables */        
			IloNumVarArray	x(env,n,0,5);
            IloNumVarArray	r(env,n,0,IloInfinity);
            
            /* Declare Constraints */
            for (int i=0; i<n; i++)
				model.add(r[i]*r[i] <= IloScalProd(A[i],x)+b[i]);
			
			/* Declare Objective */
			IloExpr objective(env);
			objective = IloScalProd(x,s[k])-IloSum(r);
			model.add(IloMinimize(env,objective));
			objective.end();
			
			/* Solve the Problem */
			IloCplex cplex(model);
			cplex.setOut(env.getNullStream());
			cplex.solve();
			
			/* Save the Results */
			mxArray *x_out = mxCreateDoubleMatrix(n, 1, mxREAL); 
			mxArray *objective_out = mxCreateDoubleMatrix(1, 1, mxREAL); 
			mxArray *diagnosis_out = mxCreateDoubleMatrix(1, 1, mxREAL); 
			double* out_x = mxGetPr(x_out);
			double* out_objective = mxGetPr(objective_out);
			double* out_diagnosis = mxGetPr(diagnosis_out);
			*out_objective = cplex.getObjValue();
			*out_diagnosis = cplex.getStatus();
			for (int i=0; i <n; i++)
				out_x[i] = cplex.getValue(x[i]);
			mxSetField(optimal, k, "x", x_out);
			mxSetField(optimal, k, "objective", objective_out);
			mxSetField(optimal, k, "diagnosis", diagnosis_out);
			
			/* Close CPLEX and Model & Free Memory! */
			cplex.end();
			model.end();
		}
    }
    catch (IloException& ex) {
        mexErrMsgTxt ("Error Stoc");
    }
    catch (...) {
        mexErrMsgTxt ("Error");
    }
    env.end();
    
    return;
}