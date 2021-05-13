'''
Build TW-verifer and reach model
In reach model using Bfs tranverse function
builder:shanghengyu
'''

import copy
from itertools import product
import time

class Reach:
    k1 = 0
    k2 = 0
    # tw intial stateQtw0,total state Qtw ,transition function t_trans
    #tw observable event t_inputo, tw unobservable event t_inputu,final state=counter
    #reach state stored the reach model's state;
    # counterInverse count the transition state which have the reflexity

    Qtw0 = []
    Qtw = []
    t_trans = {}
    t_inputo = []
    t_inputuo = []
    final_state = {}

    reachState = []
    counterInverse = {}
    newTrans = {}

    #give the value of each components and construct counter
    def __init__(self, state, initial_State, tw_trans, t_inputo, t_inputu, k1, k2):
        self.Qtw0 = initial_State.copy()
        self.Qtw = state.copy()
        self.t_trans = copy.deepcopy(tw_trans)
        self.t_inputo = t_inputo.copy()
        self.t_inputuo = t_inputu.copy()
        self.k1 = k1
        self.k2 = k2

        flag = 0
        for state in initial_State:
            flag = flag + 1
            self.final_state[(0, 0, flag)] = [state]  # count fist:k1 second:k2 third:path_number

    #start the tranverse
    def start(self):
        # eliminate the reflexive transfer function
        print("t_trans",self.t_trans)
        self.counterInverse, self.newTrans = self.countInverse(self.t_trans)
        # Implement path counter through breadth traversal
        final_state = self.tree(self.final_state)

    # Reflect the state in the counter and update the counter value of the counter
    def tree(self, final_state):  # {(0, 0, 1): [('0', '1', '3', '4')]}
        counterAndState = copy.deepcopy(final_state)
        transFunction = copy.deepcopy(self.newTrans)
        flag = 1
        counterAndState = self.checkInverse(counterAndState)
        while (flag ==1):
            currentReachState = []
            oldCounter = copy.deepcopy(counterAndState)
            counterAndState, currentReachState = self.checkTrans(counterAndState, transFunction)
            newC = copy.deepcopy(counterAndState)
            for c in newC:
                if c[0]>=k1 and c[1]>=k2:
                    state1 = self.check(counterAndState[c][-1])
                    self.reachState = list(set(self.reachState + self.check(counterAndState[c][-1])))
                    del counterAndState[c]
            for state in currentReachState:
                if state not in self.reachState:
                    self.reachState.append(state)
            transFunction = self.removeReachState(transFunction)
            if counterAndState == oldCounter:
                break
            if self.detect(self.reachState) == False:
                break
        self.detectFinal(self.reachState)

    # Extend the state that can be reached by the current state, restart the counter, and calculate the arrival state
    #teach counter that find the next tranverse function and extend the new state in the counter
    #using the Brede first search
    def checkTrans(self, counterAndState, transFunction):
        countPath = 0
        newCounter = {}
        reach_states = []
        for counter in counterAndState:  # (100, 100, 1)
            orignalSet = counterAndState[counter]
            current = orignalSet[-1]
            if current not in self.reachState:
                if counter[0] >= k1 and counter[1] >= k2:
                    currentReachState = []
                    currentReachState = self.check(current)
                    for state in currentReachState:
                        if state not in reach_states:
                            reach_states.append(state)
                for event in transFunction[current]:  # (a,a,e,e)
                    countk1 = counter[0]
                    countk2 = counter[1]
                    if event[0] in t_inputo and countk1 != 1000:
                        countk1 = countk1 + 1
                    if event[2] in t_inputo and countk2 != 1000:
                        countk2 = countk2 + 1
                    for newState in transFunction[current][event]:
                        if newState in orignalSet:    #cut the ring
                            countk1,countk2 = self.find(orignalSet,newState)
                            if countk1 >= k1 and countk2 >= k2:
                                currentReachState = []
                                currentReachState = self.check(newState)
                                for state in currentReachState:
                                    if state not in reach_states:
                                        reach_states.append(state)
                            continue
                        elif countk1 >= k1 and countk2 >= k2:
                            currentReachState = []
                            currentReachState = self.check(newState)
                            for state in currentReachState:
                                if state not in reach_states:
                                    reach_states.append(state)
                        else:
                            # for state in transFunction[current][event]:  # reach state
                            countPath = countPath + 1
                            newCounter[(countk1, countk2, countPath)] = orignalSet + [newState]
                            newCounter = self.checkInverse(newCounter)

        return newCounter, reach_states

    #count whether it exist circle
    def find(self,path, state1):
        #print("path state!!",path,state1)
        flag = path.index(state1)  # 1
        stateSet = path[flag:len(path)]  # [('1', '1', '2', '2'), ('1', '1', '3', '3'),('1', '1', '2', '2')]
        stateSet.append(state1)
        # print(stateSet)
        counterk1 = 0
        counterk2 = 0
        flag1 = 0
        while (flag1 < (len(stateSet) - 1)):
            state3 = stateSet[flag1]
            #print("state3", state3)
            for event in self.t_trans[state3]:
                for state4 in self.t_trans[state3][event]:
                    if state4 == stateSet[flag1 + 1]:
                        #print("event", event)
                        if event[0] in self.t_inputo:
                            counterk1 = counterk1 + 1
                        if event[2] in self.t_inputo:
                            counterk2 = counterk2 + 1
            flag1 = flag1 + 1
            if counterk1 !=0:
                counterk1 =1000
            if counterk2 !=0:
                counterk2 = 1000
        return counterk1, counterk2

    #check reach state which is not meet the verification method. if not which will return false.
    def detect(self, reachStates):
        failReach = []
        for state in reachStates:
            if state[0] == state[2] and state[1] == state[3]:
                if state[0] != state[1]:
                    failReach.append(state)
        if failReach != []:
            print("not ", k1, "-", k2, "detactability, because", failReach, " can not meet the requirement")
            return False

    #check whether reach state's commet the requirement
    def detectFinal(self, reachStates):
        flag = 0
        print("reachStates are", reachStates)
        failReach = []
        if reachStates == []:
            print("not ", k1, "-", k2, "detactability, because none states can reach")
        else:
            for state in reachStates:
                if state[0] == state[2] and state[1] == state[3]:
                    # print(state)
                    if state[0] == state[1]:
                        flag = flag + 1
                        # print("reach",state)
                    if state[0] != state[1]:
                        failReach.append(state)
                else:
                    flag = flag + 1
            if failReach != []:
                print("not ", k1, "-", k2, "detactability, because", failReach, " can not meet the requirement")
                return False
            if flag == len(reachStates):
                print("it is", k1, "-", k2, "detectability")

    #remove reach state's transition function
    def removeReachState(self, transFunction):
        for state in self.reachState:  # 去除到达状态的传递函数
            transFunction[state] = {}
        newTransFunction = transFunction.copy()
        for state in transFunction:  # 去除传递函数中到达reachstate的状态
            for event in transFunction[state]:
                for current in transFunction[state][event]:
                    if current in self.reachState:
                        newTransFunction[state][event].remove(current)

        for state in transFunction:  # 当传递函数中没有到达值，去除所有事件
            flag = len(transFunction[state])
            nullCount = 0
            for event in transFunction[state]:
                if transFunction[state][event] == []:
                    nullCount = nullCount + 1
            if nullCount == flag:
                newTransFunction[state] = {}


        return newTransFunction

    #find the reach states that is not in the reach state and input
    def check(self, current):  # {(0,0):[('1','2','3','4')],(1,1):[('2','0','3','4')]}
        reach_states = []
        States = self.reach_state(current)
        for state in States:
            if state not in reach_states:
                reach_states.append(state)
        return reach_states

    # find reach states next states and put them all in the reach states set
    def reach_state(self, state):
        reach_states = []
        reach_states.append(state)
        m = 1
        while (m == 1):
            key1 = len(reach_states)
            for current in reach_states:
                for n in self.newTrans[current]:
                    for m in self.newTrans[current][n]:
                        if m not in reach_states:
                            reach_states.append(m)
            if len(reach_states) == key1:
                break
        # print(reach_states)
        return reach_states

    # Search again if the state is reflexive, counterAndState updates the counter
    def checkInverse(self, counterAndState):  # ('1','2','3','4'),(0,0),
        newCounter = {}
        for counter in counterAndState:
            counterk1 = counter[0]
            counterk2 = counter[1]
            counterPath = counter[2]
            current = counterAndState[counter][-1]
            count = self.counterInverse[current]
            if count[0] == 1000:
                counterk1 = count[0]
            if count[1] == 1000:
                counterk2 = count[1]
            newCounter[(counterk1, counterk2, counterPath)] = counterAndState[counter]

        return newCounter

    # transFunction delete the reflexive state
    def countInverse(self, t_trans):
        print("k1",k1,"k2",k2)
        print("k1", self.k1, "k2", self.k2)
        for state in t_trans:
            countk1 = 0
            countk2 = 0
            for event in t_trans[state]:
                for rstate in t_trans[state][event]:
                    if rstate == state:
                        if event[0] in t_inputo:
                            countk1 = 1000
                        if event[2] in t_inputo:
                            countk2 = 1000
                        t_trans[state][event].remove(rstate)
            self.counterInverse[state] = (countk1, countk2)
        return self.counterInverse, t_trans

#build the TW-verifier's class
class TW_verifier:

    tw_states = set()  # All state sets Qtw
    tw_initial_states = set()  # Initial stateQtw0
    tw_events = set()  # All event setEtw
    tw_trans = {}  # Transfer function, dictionary, including revelation state, character and destination
    # {'q1':{'a':'q2', 'b':'q3'}}，
    tw_final_states = set()  # final state

    #give the value for each components in TW-verifier
    def __init__(self, states, input_symbols, t_trans, start_state, t_inputo, t_inputu):

        self.tw_states = self.construct_tw_state(states).copy()
        self.tw_initial_states = self.construct_tw_initial_state(start_state, states).copy()
        self.tw_trans = copy.deepcopy(self.construct_tw_trans(t_trans, self.tw_states, t_inputo, t_inputu))
        self.tw_trans, self.tw_states, self.tw_events = self.cutState(self.tw_trans)

    #eliminate the state that it was not appear in the transition funtcion
    def cutState(self,tw_trans):
        newTransFunction = copy.deepcopy(tw_trans)
        newTransFunction = self.countInverse(newTransFunction)
        allState = []
        allState = allState + self.tw_initial_states
        allEvent = []
        flag = 1
        newState = []
        newState = newState + self.tw_initial_states
        while(flag==1):
            allState,allEvent,newState = self.subCutState(newState,allState,allEvent,newTransFunction)
            if newState == []:
                break
        newTrans = copy.deepcopy(tw_trans)
        for state in tw_trans:
            if state not in allState:
                del newTrans[state]
        return newTrans,allState,allEvent

    #to traverse the transition function from initial to the end
    def subCutState(self,newstate,states,events,newTransFunction):
        newState1 = []
        for state in newstate:
            for event in newTransFunction[state]:
                if event not in events:
                    events.append(event)
                for addstate in newTransFunction[state][event]:
                    if addstate not in states:
                        states.append(addstate)
                        newState1.append(addstate)
        return states,events,newState1

    # transFunction deletes the reflexive state
    def countInverse(self, t_trans):
        for state in t_trans:
            for event in t_trans[state]:
                for rstate in t_trans[state][event]:
                    if rstate == state:
                        t_trans[state][event].remove(rstate)
        return t_trans

   #construct the total states
    def construct_tw_state(self, states):
        Qtw = []  # states Qtw
        Qtw = Carproduct(states, states)
        Qtw = Carproduct(Qtw, states)
        Qtw = Carproduct(Qtw, states)
        # print(len(Qtw))
        return Qtw

    # construct the initial states
    def construct_tw_initial_state(self, initial_state, states):
        Qtw0 = []
        Qtw0 = Carproduct(initial_state, initial_state)
        Qtw0 = Carproduct(Qtw0, states)
        Qtw0 = Carproduct(Qtw0, states)
        # print(len(Qtw0))
        return Qtw0

    # construct the transition function
    def construct_tw_trans(self, t_trans, Qtw, t_inputo, t_inputu):
        # Construct a transfer function for each state, that is,
        # input different input symbols for each state to construct a transfer function
        new_trans = {}
        # Disregarding non-directional automata
        for q in Qtw:  # q state ('3', '2', '3', '3')
            new_subtrans = {}
            # s input symbol a
            for s in t_inputo:  # "0":{"u": ["2"]}
                new_states = []
                if s in t_trans[q[0]].keys() and s in t_trans[q[1]].keys():  # 考虑x1 x2
                    for m in t_trans[q[0]][s]:
                        for n in t_trans[q[1]][s]:
                            new_states.append((m, n, q[2], q[3]))
                    new_subtrans[(s, s, 'e', 'e')] = new_states
                    new_states = []
                # consider x3，x4   {'x':{'a':['0']}}
                # x3 = x4 = -1
                for x in t_trans.keys():  # Traverse each state, first see if x3 is suitable
                    if s in t_trans[x].keys():  # Find out the state transitions in each state,
                                                # and judge whether the current symbol can be input in the current state,
                        if q[2] in t_trans[x][s]:
                            # x3 = x
                            for y in t_trans.keys():  # After finding a suitable one for x3, traverse x4 again to see if there is a suitable one
                                if s in t_trans[y].keys():
                                    if q[3] in t_trans[y][s]:
                                        # x4 = y
                                        newState = ((q[0], q[1], x, y))
                                        new_states.append(newState)
                            # x3 = x4 = -1
                if new_states != []:
                    new_subtrans[('e', 'e', s, s)] = new_states
                    new_states = []
            # input symple u
            for s in t_inputu:
                new_states = []
                if s in t_trans[q[0]].keys():  # x1
                    for m in t_trans[q[0]][s]:
                        new_states.append((m, q[1], q[2], q[3]))
                    new_subtrans[(s, 'e', 'e', 'e')] = new_states
                    new_states = []
                if s in t_trans[q[1]].keys():  # x2
                    for m in t_trans[q[1]][s]:
                        new_states.append((q[0], m, q[2], q[3]))
                    new_subtrans[('e', s, 'e', 'e')] = new_states
                    new_states = []
                # x3
                for x in t_trans.keys():  # Traverse every state
                    if s in t_trans[x].keys():  # Find out the state transitions in each state, and judge whether the current symbol can be input in the current state
                        if q[2] in t_trans[x][s]:
                            new_states.append((q[0], q[1], x, q[3]))
                if new_states != []:
                    new_subtrans[('e', 'e', s, 'e')] = new_states
                    new_states = []
                # x4
                for x in t_trans.keys():  # Traverse every state
                    if s in t_trans[x].keys():  # Find out the state transitions in each state, and judge whether the current symbol can be input in the current state
                        if q[3] in t_trans[x][s]:
                            new_states.append((q[0], q[1], q[2], x))
                if new_states != []:
                    new_subtrans[('e', 'e', 'e', s)] = new_states
                    new_states = []
            new_trans[q] = new_subtrans
        return new_trans

#dicarset product
def Carproduct(list1, list2):
    newList = []
    for x, y in product(list1, list2):
        if type(x) == str:
            x = (x,)
        if type(y) == str:
            y = (y,)
        newList.append(x + y)
    return newList

#give the value for NFA's components it need to follow the instructions to build the model
def InputNFA(t_states,t_start,t_input,t_inputu,t_inputo,t_trans):
    # input automaton's states, input_symbols, trans, start_state, final_states
    print("Now: please input your automaton(NFA) model:")
    print("when total state are 0,2,3 and you need to input 0,2,3")
    states = input("input automaton's total set:")
    t_states = states.split(",")
    print("when initial state are 0 and you need to input 0")
    start = input("input automaton's initial set:")
    t_start = start.split(",")
    print("when events are a,b,u and you need to input a,b,u")
    events = input("input automaton's total set:")
    t_input = events.split(",")
    print("when unobservable events are u and you need to input u")
    eventu = input("input automaton's total set:")
    t_inputo = eventu.split(",")
    print("When observable events are a,b and you need to input a,b")
    evento = input("input automaton's observable set:")
    t_inputu = evento.split(",")
    print("Now,input transition functions(remind that all states need have the transition function):")
    flag = len(t_states)

    for state in t_states:
        newdic = {}
        print("============transition function ", state, " =============")
        print("initial state is", state)
        flag1 = int(input("How many event accepted by this state:"))
        while (flag1 > 0):
            value1 = input("please input the event:")
            print("when reach states are 1,2 and you need to input 1,2")
            value2 = input("please input the reach states:")
            list1 = value2.split(",")
            newdic[value1] = list1
            flag1 = flag1 - 1
        t_trans[state] = newdic
        flag = flag - 1
        print("============transition function", state, " end =============")
    return t_states,t_start,t_input,t_inputu,t_inputo,t_trans

if __name__ == "__main__":

    startTime = time.perf_counter()  # caculate the running time
    t_trans={}
    t_states=[]
    t_start=[]
    t_input=[]
    t_inputu=[]
    t_inputo=[]
    t_states,t_start,t_input,t_inputu,t_inputo,t_trans=InputNFA(t_states,t_start,t_input,t_inputu,t_inputo,t_trans)

    print("This automaton(NFA) is ")
    print("all states",t_states)
    print("initial state",t_start)
    print("events",t_input)
    print("observable events",t_inputu)
    print("unoservable events",t_inputo)
    print("transition functions",t_trans)

    k1=int(input("Now please input k1"))
    k2 = int(input("Now please input k2"))

    #construct tw-verifier model model
    tw = TW_verifier(t_states, t_input, t_trans, t_start, t_inputo, t_inputu)
    #construct reach model and judge whtether NFA is (k1,k2)-detectability
    reach = Reach(tw.tw_states, tw.tw_initial_states, tw.tw_trans, t_inputo, t_inputu, k1, k2)

    reach.start()
    #caculate the end time
    end = time.perf_counter()

    print("running time", end-startTime,"s")


