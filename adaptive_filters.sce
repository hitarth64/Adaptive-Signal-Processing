// RLS2: Implements Recursive Least Square adaptive filter given lambda for decaying contribution, delta for initialization - can be anything but zero and finally, x_n the input data  // sequence.

//LMS: LMS adaptive filter - mu: adaptive weight to be given as input and x_n input signal.  

// sliding window adaptive filter: L: length of window, delta- anything but zero. 

// All these are currently set to be linear predictors. Could however be changed. 

function [w] = RLS2(lambda, delta, x_n)
    w(:,1) = [0 0]';
    P = eye(2,2) / delta;
    
    for i = 1:length(x_n)-1
        x_vec = [x_n(i) 0]';
        
        if i>1 then x_vec(2) = x_n(i-1);
        end
        
        z_n = P * x_vec;
        g_n = 1/(lambda + x_vec' * x_vec) * z_n;
        alpha = x_n(i+1) - w(:,i)' * x_vec;
        w(:,i+1) = w(:,i) + alpha*g_n;
        P = (P - g_n * z_n')/lambda;
    end
endfunction

function[w] =LMS(mu,x_n)
    w(:,1) = [0; 0];
    
    for i = 1:length(x_n)-1
        second = 0;
        
        if i>1 then second = x_n(i-1);
        end
        x_vector = [x_n(i); second];
        err=x_n(i+1) - w(:,i)'*x_vector;
        
        w(:,i+1) = w(:,i) + mu*err*x_vector;
    end
endfunction


function [w] = sliding(L,x_n, delta)
    w(:,1) = [0 0]';
    P = eye(2,2) / delta;
    
    for i = 1:length(x_n)-1
        x_vec = [x_n(i) 0]';
        
        if i>1 then x_vec(2) = x_n(i-1);
        end
        
        g_n = 1/(1 + x_vec' * P * x_vec) *P * x_vec;
        w_tilde = w(:,i) + g_n *(x_n(i+1) - w(:,i)' * x_vec);
        P_tilde = P - g_n * x_vec' * P;
        
        first=0; second=0;
        if i-L-1>0 then first = x_n(i-L-1)
        end
        if i-L-2>0 then second = x_n(i-L-2)
        end
        x_n_L_1 = [first second]';
        g_tilde = P_tilde* x_n_L_1 / (1-x_n_L_1' *P_tilde* x_n_L_1);
        
        d_n_L_1 = 0;
        if i-L-1>0 then d_n_L_1 = x_n(i-L);
        end
        
        w(:,i+1) = w_tilde - g_tilde*(d_n_L_1 - w_tilde' * x_n_L_1);
        P = P_tilde + g_tilde * x_n_L_1' * P_tilde;
    end
endfunction

