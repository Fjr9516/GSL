Guided Subspace Learning(GSL demo)
=======================================
In the file, there are multiple .m files.

>>GSL.m : Core codes of GSL algorithm.

>>NGSL.m : Core codes of NGSL algorithm.

>>demoGSL.m : Evaluate GSL/NGSL on an example task (C-W in 4DA dataset).

You can run the "demoGSL.m" code for your reference.  Then you will get the results:

>>GSL: Choose 'primal' as 'kernel_type', the final accrucy will be ‘55.93%’.

>>NGSL: Choose 'linear'(linear kernel in our paper,for example) as 'kernel_type' , the final accrucy will be ‘63.39%’.

When you use our code, there are two(GSL)/three(NGSL) main parameters need to be adjusted according to different tasks:
>>alpha,beta and lambda 

Once you run the code, please correctly set the path of the data and liblinear toolbox

==========================================
Please contact leizhang@cqu.edu.cn or jrfu@cqu.edu.cn if there is any problem

Enjoy it!
