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
    #define param       prhs[0]
    #define Q_xx_in 	prhs[1]
	#define Q_xs_in 	prhs[2]
	#define q_in 		prhs[3]
    #define s_in        prhs[4]    
	#define optimal 	plhs[0]
    
    if (nrhs != 5) {
        mexErrMsgTxt("Five input arguments required.");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
	#define optimal plhs[0]
    mxArray *h_in       = mxGetField(param, 0, "h");
    mxArray *H_in       = mxGetField(param, 0, "H");
    mxArray *W_in       = mxGetField(param, 0, "W"); 
    mxArray *delta_in   = mxGetField(param, 0, "delta");
    
    size_t n	= mxGetNumberOfElements(q_in);
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
		NumMatrix 	Q_xx(env,n);
		NumMatrix 	Q_xs(env,m);
		IloNumArray q(env,n);    
		NumMatrix 	W(env,nh);
        NumMatrix 	H(env,nh);
        IloNumArray h(env,nh);
        IloNum 		delta;		
		for (int i=0; i < N; i++) {
			s[i] = IloNumArray(env,m);
			for (int j=0; j<m; j++)
                s[i][j] = mxGetPr(s_in)[j+i*m];
        }   
		for (int i=0; i<n; i++){
			Q_xx[i] = IloNumArray(env,n);
			q[i] = mxGetPr(q_in)[i];
			for (int j=0; j<n; j++)
				Q_xx[i][j] = mxGetPr(Q_xx_in)[j+i*n];
		}
		for (int i=0; i<m; i++){
			Q_xs[i] = IloNumArray(env,n);
			for (int j=0; j<n; j++)
				Q_xs[i][j] = mxGetPr(Q_xs_in)[j+i*n];
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
        delta = mxGetScalar(delta_in);

		
		for (int k=0; k<N; k++){	
			/* The first model */	
			IloModel model1(env);
			
			/* Define Decision Variables */        
			IloNumVarArray	x1(env,n,-IloInfinity,IloInfinity);
			
			/* Declare Constraints */
			for (int i=0; i<nh; i++)
				model1.add(IloScalProd(W[i],x1) >= IloScalProd(H[i],s[k]) + h[i]);
			
			/* Declare Objective */
			IloExpr objective1(env);
			for (int i=0; i<n; i++)
				for (int j=0; j<n; j++)
					objective1 += x1[i]*Q_xx[i][j]*x1[j];
			for (int i=0; i<m; i++)
				for (int j=0; j<n; j++)
					objective1 += s[k][i]*Q_xs[i][j]*x1[j];
			for (int i=0; i<n; i++)
				objective1 += q[i]*x1[i];
			model1.add(IloMinimize(env,objective1));
			objective1.end();
			
			/* Solve the Problem */
			IloCplex cplex1(model1);
			cplex1.setOut(env.getNullStream());
			cplex1.solve();

			/* Save first model objective value */
			double obj1 = cplex1.getObjValue();

			/* Close CPLEX and Model & Free Memory! */
			cplex1.end();
			model1.end();

			/* The second model */
			IloModel model2(env);
			
			/* Define Decision Variables */        
			IloNumVarArray	x2(env,n,-IloInfinity,IloInfinity);
			
			/* Declare Constraints */
			for (int i=0; i<nh; i++)
				model2.add(IloScalProd(W[i],x2) >= IloScalProd(H[i],s[k]) + h[i]);
			IloExpr subopt(env);
			for (int i=0; i<n; i++)
				for (int j=0; j<n; j++)
					subopt += x2[i]*Q_xx[i][j]*x2[j];
			for (int i=0; i<m; i++)
				for (int j=0; j<n; j++)
					subopt += s[k][i]*Q_xs[i][j]*x2[j];
			for (int i=0; i<n; i++)
				subopt += q[i]*x2[i];
			double r = ((double) rand() / (RAND_MAX));
            model2.add(subopt <= obj1 + (delta * r));
            subopt.end();

			/* Declare Objective */
            IloExpr objective2(env);
            for (int i=0; i<n; i++)
				objective2 += x2[i]*((double) rand() / (RAND_MAX))*x2[i];
			for (int i=0; i<n; i++)
				objective2 += -((double) rand() / (RAND_MAX))*x2[i];
			model2.add(IloMinimize(env,objective2));
			objective2.end();
			
			/* Solve the Problem */
			IloCplex cplex2(model2);
			cplex2.setOut(env.getNullStream());
			cplex2.solve();
			
			/* Save the Results */
			mxArray *x_out = mxCreateDoubleMatrix(n, 1, mxREAL); 
			mxArray *objective_out = mxCreateDoubleMatrix(1, 1, mxREAL); 
			mxArray *diagnosis_out = mxCreateDoubleMatrix(1, 1, mxREAL); 
			double* out_x = mxGetPr(x_out);
			double* out_objective = mxGetPr(objective_out);
			double* out_diagnosis = mxGetPr(diagnosis_out);
			*out_objective = cplex2.getObjValue();
			*out_diagnosis = cplex2.getStatus();
			for (int i=0; i <n; i++)
				out_x[i] = cplex2.getValue(x2[i]);
			mxSetField(optimal, k, "x", x_out);
			mxSetField(optimal, k, "objective", objective_out);
			mxSetField(optimal, k, "diagnosis", diagnosis_out);
			
			/* Close CPLEX and Model & Free Memory! */
			cplex2.end();
			model2.end();
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