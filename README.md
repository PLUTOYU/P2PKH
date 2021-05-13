Two programs are used to verify whether the automaton is (k1,k2)-detectability.  
1.To input the automaton, the first step is to change the automaton’s four components into code structure.  
(1)Input the total state, the format needs be state1,state2,state3...  
(2)Input the initial state, the format needs to be state1,state2,state3...  
(3)Input the total events, the format needs to be event1,event2,event3...  
(4)Input the observable events,the format needs to be event1,event2,event3...  
(5)Input the unobservable events,the format needs to be event1,event2,event3...  
(6)As for the transition function,it is the dictionary, the program will iterate all states in automaton, and you could input the corresponding transition event and reach state.  
In each state, the program will ask u how many events accepted by this state. And then you need to input each event and the reach state. When this state with the event will lead automaton into many reach states, it need to input state1,state2  
2.After input automaton, it need to input the k1,k2’s values.  
3.The system will running to give the result.  

For example:  
Now: please input your automaton(NFA) model:  
when total state are 0,2,3 and you need to input 0,2,3  
input automaton's total set:0,2,3  
when initial state are 0 and you need to input 0  
input automaton's initial set:0  
when events are a,b,u and you need to input a,b,u  
input automaton's total events:a,b,u  
when unobservable events are u and you need to input u  
input automaton's unobservable events:u  
When observable events are a,b and you need to input a,b  
input automaton's observable events:a,b  
Now,input transition functions(remind that all states need have the transition function):  
============transition function  0  =============  
initial state is 0  
How many event accepted by this state:2  
please input the event:a  
when reach states are 1,2 and you need to input 1,2  
please input the reach states:3  
please input the event:u  
when reach states are 1,2 and you need to input 1,2  
please input the reach states:2  
============transition function 0  end =============  
============transition function  2  =============  
initial state is 2  
How many event accepted by this state:1  
please input the event:a  
when reach states are 1,2 and you need to input 1,2  
please input the reach states:3  
============transition function 2  end =============  
============transition function  3  =============  
initial state is 3  
How many event accepted by this state:1  
please input the event:b  
when reach states are 1,2 and you need to input 1,2  
please input the reach states:3  
============transition function 3  end =============  
==========This automaton(NFA) is==========   
all states ['0', '2', '3']  
initial state ['0']  
events ['a', 'b', 'u']  
observable events ['u']  
unoservable events ['a', 'b']  
transition functions {'0': {'a': ['3'], 'u': ['2']}, '2': {'a': ['3']}, '3': {'b': ['3']}}  
=============================================   
Now please input k1: 1  
Now please input k2: 1  
To verify the 1 - 1  delayed detectability  
=============================================  
reachStates are [('3', '3', '0', '0'), ('3', '3', '2', '0'), ('3', '3', '0', '2'), ('3', '3', '2', '2'), ('3', '3', '3', '3')]  
it is 1 - 1 detectability  
running time 42.1814758 s  
