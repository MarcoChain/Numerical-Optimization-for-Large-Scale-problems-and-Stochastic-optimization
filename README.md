# Interior Point Method applied to linear programming problems
  
### Introduction
Let's start with the definition of an Interior point method: an Interior point method is an algorithm that is used in solving both linear and nonlinear convex optimization problems that contain inequalities as constraints. The aim of this research is to implement the algorithm for a linear programming problem which it'll be discussed in the next section, the programming language used for this assignment is MATLAB. 

We consider the linear programming problem in standard form: 
$$
 min\ f(x) \\ 
 s.t.\ h(x) = 0,\ g(x) \leq 0
$$
where $$f(x): \textbf{R}^n \rightarrow \textbf{R},\ h(x): \textbf{R}^n \rightarrow \textbf{R}^m,\ g(x): \textbf{R}^n \rightarrow \textbf{R}^r$$

In order to solve the problem we start computing its optimality conditions. \newline
Optimality conditions can be derived from the lagrangian function theory. By introducing the lagrangian function for the equality constraint $h_i(x)$
$$
\mathcal{L}(x, \lambda) = f(x) + \lambda_i h_i(x)
$$
and noting that $\nabla_{x}\mathcal{L}(x, \lambda) = \nabla f(x) + \lambda \nabla h(x)$, we observe that at the solution $x^{*}$, there exists a scalar $\lambda_i ^{*}$ such that 
$$
\nabla_{x}\mathcal{L}(x^*, \lambda_i^{*}) =0.
$$
This observation suggests that we can search for solutions of the equality-constrained
problem by looking for stationary points of the Lagrangian function. The scalar $\lambda_i$  is called a Lagrange multiplier vector for the constraint $h_i(x)$. 

Considering all the constraints, the Lagrangian function is the following:
$$
\mathcal{L}(x, \lambda, \mu) = f(x) + \sum_{i=1}^{n} \lambda_i h_i(x) + \sum_{j=1}^{r} \mu_j g_j(x)
$$
The necessary conditions defined below are called first-order conditions because they are concerned with properties of the gradients (first-derivative vectors) of the objective and constraint functions. These conditions are the foundation for many algorithms concerning numerical optimization, also Interior point method.

Before defining the solution of the system, we introduce the concept of the active set. The active set is the set of indices where the inequality constraints are active. 
$$
\mathcal{A}(x) = \{ j \in {1,\dotsc,r}: g_j(x)=0 \}
$$

Suppose that $x^{*}$ is a local solution of the general optimization problem, that the functions $f(x),\ h(x),\ g(x)$ are continuously differentiable. Then there exist  $\lambda^{*}, \mu^{*}$, such that:
$$
\nabla_{x}\mathcal{L}(x^*, \lambda^{*}, \mu^*) =0 \\
\nabla_{\lambda}\mathcal{L}(x^*, \lambda^{*}, \mu^*) =0 \\
\mu_j \geq 0,\ \forall j \in {1,\dotsc,r} \\
\mu_j = 0,\ \forall j \notin \mathcal{A}(x^*) 
$$

These are called first-order conditions or Karush–Kuhn–Tucker (KKT) conditions. Convexity of the problem ensures that these conditions are sufficient for a global
minimum. 

To look at the details and all the algorithms implemented looks at  our [tesina](https://github.com/MarcoChain/Numerical-Optimization-for-Large-Scale-problems-and-Stochastic-optimization/blob/master/assignment_opt_c1.pdf)
# Feed forward neural network implementation
There are several ways in which it is possible to implement a non-linear regressor, but a backpropagation method seems what it is needed to solve this problem. Backpropagation needs for every possible entry tuple an exit value in order to compute the loss function. It is then minimized with a gradient-method as it’s shown in the least-squared approach. This is the basic idea behind neural network (NN). These kind of networks are able to optimally recognize various types of patterns, and they are used in different fields such as regression. However, it is necessary to tune several hyperparameters to obtain an adequate result. NN are essentially based on nodes: 
- Input nodes, which are in finite number, that usually equals the number of inputs of the function. Our problem, for example, needs four nodes, because we have PM10, humidity, temperature and atmospheric pressure as inputs. The set of all input nodes constitutes the so-called input layer. 

- Hidden nodes makem up that part of the NN that introduces a non-linearity into the model. They are organized in one or more layers. There are no fixed rules to select the correct number of hidden nodes or layers: only a "trial and error" approach can find an appropriate solution to problems (for our purpose, one hidden layer is enough but the problem of the number of nodes remains).
-   Output node, which is one and only one for a regression task. NN can acquire more than 8 one output node, but in these cases they are used in other situations and fields which are not relevant in our situation.

It's important to emphasize that every node of the input node is fully linked with the nodes of the hidden layer and, in turn, every node of hidden layer is connected with the output node. Each connection is only in one direction and it's associated with a scalar value called weight. From now on, I will denote by $w(i,h)$the weight which joins the $Ith$  input layer with the $Hth$ hidden one. Indeed, I will denote by $x(h)$ the weight on the link from the $Hth$ hidden node to the output node. So, every node has an internally stored value: for the input layer, we have the input values of dataset, while for the hidden and the output node we can compute this value using $w$ and $x$. we denote by $u(j)$ the value of the  $jth$  input node, we can compute the vector $v$ as:

$$ 
\hat{v}(j) = \sum_{j=1}^I w(i,j)u(j)
$$
This doesn't exactly match the value of hidden nodes, because we have to do a sort of  'normalization' using a function where the result is a number in the range between $[0, 1]$. There are different functions that can be used, but the most popular ones are the sigmoid and the ReLU function. These two are the only ones that work with the data we have available.
Computed the value  v of each hidden node, the output value of the NN is given by:

$$ o = \sum_{h=1}^{H} x(h)v(h)$$

At this point, the backpropagation algorithm is used to minimize the squared differences between actual function values and predicted values. At this point, if I denote by $e_p$ the quantity:
$$ e_p = y_p - o_p $$
I can determinate the normalized mean sum of squared error (SSE) as:

$$SSE_p = \sum_{p=1}^{n}(y_p - o_p)^2 $$

At last, the  backpropagation algorithm,  like any other steepest-descent method,  minimize $SSE/2$. If $SSE/2$ is minimized, $SSE$ will be minimized too. 
Finally, we can update the values of $w$ and $x$as:

$$
w(i,h) \leftarrow w(i,h) + \mu \sum_{p=1}^{n} (y_p - o_p)x(h)v_p(h)(1-v_p(h))u(i)
$$ $$
x(h) \leftarrow  x(h) + \mu \sum_{p=1}^{n} (y_p - o_p)v_p(h)
$$

Where $\mu$ is a scalar value also called $\textbf{ learning rate}$. This value is one of the most important in all the NN. The hyperparameter  $\mu$ can heavily influence convergence and the speed of our network. 
This process is repeated a number of times which is called $\textbf{Epochs}$ and this parameter is also chosen by the user. If the number of epochs is too large, the model can have the so-called $\textbf{overfitting}$ problem. To avoid this problem it's possible to choose a tolerance: in this way the algorithm only stops when the difference:
$$
 SSE_{new} - SSE_{old} <= tolerance 
$$

To look at the details and all the algorithms implemented looks at  our [tesina](https://github.com/MarcoChain/Numerical-Optimization-for-Large-Scale-problems-and-Stochastic-optimization/blob/master/Relazione.pdf)

> Written by [MarcoChain](https://www.linkedin.com/in/marcogullotto/).
