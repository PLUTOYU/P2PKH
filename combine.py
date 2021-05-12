import copy
from itertools import product
import time
#深度遍li



class Reach:
    k1 = 0
    k2 = 0

    Qtw0 = []
    Qtw = []
    t_trans = {}
    t_inputo = []
    t_inputuo = []
    final_state = {}

    reachState = []
    counterInverse = {}
    newTrans = {}
    newTransFunction = {} #存放迁移函数中状态变为计数器的集合

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


    def start(self):
        # 首先消去自反传递函数
        self.counterInverse, self.newTrans = self.countInverse(self.t_trans)
        #print("t_trans",self.newTrans)

        #通过深度遍历实现路径的计数器
        final_state = self.tree(self.final_state)



    # 将计数器中的状态进行自反，更新计数器的counter值 构造一个栈stack
    def tree(self, final_state):  # {(0, 0, 1): [('0', '1', '3', '4')]}
        counterAndState = copy.deepcopy(final_state)
        transFunction = copy.deepcopy(self.newTrans)
        flag = 1
        #检测自反 记录自反
        counterAndState = self.checkInverse(counterAndState)
        transFunction = self.changeTrans(transFunction)

        flag=0

        #state1 = stack.pop
        while(counterAndState!={} ):
            subCounterAndState = counterAndState.popitem()
            #print("counterAndState",counterAndState)
            if subCounterAndState[1][-1] not in self.reachState:
                counterAndState,transFunction=self.checkTrans(counterAndState,subCounterAndState,transFunction)
            transFunction = self.removeReachState(transFunction)
            if self.detect(self.reachState) == False:
                break

        self.detectFinal(self.reachState)



    def checkTrans(self,counterAndState,subCounterAndState,transFunction):
        count1 = subCounterAndState[0][0]
        count2 = subCounterAndState[0][1]
        count3 = subCounterAndState[0][2] - 1
        if count1 >= k1 and count2 >=k2:
            currentReachState = []
            currentReachState = self.check(subCounterAndState[1][-1])
            for state in currentReachState:
                if state not in self.reachState:
                    self.reachState.append(state)
            return counterAndState,transFunction
        for event in transFunction[subCounterAndState[1][-1]]:
            count1 = count1 + event[0]
            count2 = count2 + event[1]
            if subCounterAndState[0][0] >= k1 and subCounterAndState[0][1] >= k2:
                for reState in transFunction[subCounterAndState[1][-1]][event]:
                    stateSet = self.check(reState)
                    for state1 in stateSet:
                        if state1 not in self.reachState:
                            self.reachState.append(state1)
            else:
                for state in transFunction[subCounterAndState[1][-1]][event]:
                    if state in subCounterAndState[1]:  # 去环
                        countk1, countk2 = self.find(subCounterAndState[1], state)
                        if countk1 >= k1 and countk2 >= k2:
                            currentReachState = []
                            currentReachState = self.check(state)
                            for state2 in currentReachState:
                                if state not in self.reachState:
                                    self.reachState.append(state)
                    else:
                        count3 = count3 + 1
                        counterAndState[(subCounterAndState[0][0], subCounterAndState[0][1], count3)] = \
                        subCounterAndState[1] + [state]
                        counterAndState = self.checkInverse(counterAndState)

        return counterAndState,transFunction

    def changeTrans(self,transFunction):
        newTransFaunction = {}
        for state in transFunction:
            newTrans = {}
            for event in transFunction[state]:
                counterk1 = 0
                counterk2 = 0
                if event[0] in t_inputo:
                    counterk1 = 1
                if event[2] in t_inputo:
                    counterk2 = 1
                newCounter = (counterk1,counterk2)
                if newCounter not in newTrans.keys():
                    newTrans[newCounter] = transFunction[state][event]
                if newCounter in newTrans.keys():
                    newTrans[newCounter] = newTrans[newCounter] + transFunction[state][event]
            newTransFaunction[state] = newTrans
        return newTransFaunction

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

    def detect(self, reachStates):
        failReach = []
        for state in reachStates:
            if state[0] == state[2] and state[1] == state[3]:
                if state[0] != state[1]:
                    failReach.append(state)
        if failReach != []:
            print("not ", k1, "-", k2, "detactability, because", failReach, " can not meet the requirement")
            return False


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



    def check(self, current):  # {(0,0):[('1','2','3','4')],(1,1):[('2','0','3','4')]}
        reach_states = []
        # if countk1 >= k1 and countk2 >=k2:
        # print("reach")
        # print(current)
        States = self.reach_state(current)
        for state in States:
            if state not in reach_states:
                reach_states.append(state)
        # print(reachStates)
        return reach_states

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

    # 搜索一遍状态是否有自反,counterAndState更新计数器
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

    # transFunction将自反状态删除
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


class TW_verifier:
    tw_states = set()  # 所有状态集Qtw
    tw_initial_states = set()  # 初始状态Qtw0
    tw_events = set()  # 所有事件集Etw
    tw_trans = {}  # 转移函数，字典，包括启示状态，字符与终点{'q1':{'a':'q2', 'b':'q3'}}，NFA包括e转移规则 ftw
    tw_final_states = set()  # 终结状态

    def __init__(self, states, input_symbols, t_trans, start_state, t_inputo, t_inputu):

        self.tw_states = self.construct_tw_state(states).copy()
        self.tw_initial_states = self.construct_tw_initial_state(start_state, states).copy()

        #print("initialstate",self.tw_initial_states)

        #self.tw_events = self.construct_tw_events(input_symbols).copy()
        #print("oldevent",len(self.tw_events))
        #print("oldevent",self.tw_events)

        self.tw_trans = copy.deepcopy(self.construct_tw_trans(t_trans, self.tw_states, t_inputo, t_inputu))


        self.tw_trans, self.tw_states, self.tw_events = self.cutState(self.tw_trans)


    def cutState(self,tw_trans):
        newTransFunction = copy.deepcopy(tw_trans)
        #print("transfunction", newTransFunction)
        newTransFunction = self.countInverse(newTransFunction)
        #print("newtransfunction",newTransFunction)
        allState = []
        allState = allState + self.tw_initial_states
        allEvent = []
        flag = 1
        newState = []
        newState = newState + self.tw_initial_states
        while(flag==1):
            allState,allEvent,newState = self.subCutState(newState,allState,allEvent,newTransFunction)
            #print("newState",newState)
            if newState == []:
                break
        newTrans = copy.deepcopy(tw_trans)
        for state in tw_trans:
            if state not in allState:
                del newTrans[state]
        return newTrans,allState,allEvent

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

    # transFunction将自反状态删除
    def countInverse(self, t_trans):
        for state in t_trans:
            for event in t_trans[state]:
                for rstate in t_trans[state][event]:
                    if rstate == state:
                        t_trans[state][event].remove(rstate)
        return t_trans

    def construct_tw_state(self, states):
        Qtw = []  # states Qtw
        Qtw = Carproduct(states, states)
        Qtw = Carproduct(Qtw, states)
        Qtw = Carproduct(Qtw, states)
        # print(len(Qtw))
        return Qtw

    def construct_tw_initial_state(self, initial_state, states):
        Qtw0 = []
        Qtw0 = Carproduct(initial_state, initial_state)
        Qtw0 = Carproduct(Qtw0, states)
        Qtw0 = Carproduct(Qtw0, states)
        # print(len(Qtw0))
        return Qtw0


    def construct_tw_trans(self, t_trans, Qtw, t_inputo, t_inputu):
        # 对每个状态构造传递函数 即对每个状态输入不同的输入符号，使其构造传递函数
        new_trans = {}
        # 先不考虑不定向自动机
        # Qtw=[('0','1','3','4')]
        for q in Qtw:  # q状态 ('3', '2', '3', '3')
            new_subtrans = {}
            # s输入的符号 a
            for s in t_inputo:  # "0":{"u": ["2"]}
                new_states = []
                if s in t_trans[q[0]].keys() and s in t_trans[q[1]].keys():  # 考虑x1 x2
                    for m in t_trans[q[0]][s]:
                        for n in t_trans[q[1]][s]:
                            new_states.append((m, n, q[2], q[3]))
                    new_subtrans[(s, s, 'e', 'e')] = new_states
                    new_states = []
                # 考虑x3，x4   {'x':{'a':['0']}}
                # x3 = x4 = -1
                for x in t_trans.keys():  # 遍历每一个状态，首先看x3有没有合适的
                    if s in t_trans[x].keys():  # 查找出每一个状态中拥有的状态转移，并且判断当前状态中是否可以输入当前的符号,
                        if q[2] in t_trans[x][s]:
                            # x3 = x
                            for y in t_trans.keys():  # 找到x3合适的后，再遍历x4看是否有合适的
                                if s in t_trans[y].keys():
                                    if q[3] in t_trans[y][s]:
                                        # x4 = y
                                        newState = ((q[0], q[1], x, y))
                                        new_states.append(newState)
                            # x3 = x4 = -1
                if new_states != []:
                    new_subtrans[('e', 'e', s, s)] = new_states
                    new_states = []
            # 输入的符号u
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
                for x in t_trans.keys():  # 遍历每一个状态
                    if s in t_trans[x].keys():  # 查找出每一个状态中拥有的状态转移，并且判断当前状态中是否可以输入当前的符号
                        if q[2] in t_trans[x][s]:
                            new_states.append((q[0], q[1], x, q[3]))
                if new_states != []:
                    new_subtrans[('e', 'e', s, 'e')] = new_states
                    new_states = []
                # x4
                for x in t_trans.keys():  # 遍历每一个状态
                    if s in t_trans[x].keys():  # 查找出每一个状态中拥有的状态转移，并且判断当前状态中是否可以输入当前的符号
                        if q[3] in t_trans[x][s]:
                            new_states.append((q[0], q[1], q[2], x))
                if new_states != []:
                    new_subtrans[('e', 'e', 'e', s)] = new_states
                    new_states = []
            new_trans[q] = new_subtrans
        return new_trans


def Carproduct(list1, list2):
    newList = []
    for x, y in product(list1, list2):
        if type(x) == str:
            x = (x,)
        if type(y) == str:
            y = (y,)
        newList.append(x + y)



    return newList



if __name__ == "__main__":


    start = time.perf_counter()





    # states, input_symbols, trans, start_state, final_states
    t_states = ["0", "2", "3"]
    t_input = ["a", "b", "u"]
    t_inputo = ["a", "b"]
    t_inputu = ["u"]
    t_trans = {"0": {"a":[ "3"],
                     "u": ["2"]},

               "2": {"a": ["3"]},
               "3": {"b": ["3"]}
               }
    t_start = ["0"]  # 先不管自动机中的初始状态




    k1=0
    k2=1

    tw = TW_verifier(t_states, t_input, t_trans, t_start, t_inputo, t_inputu)

    print("Qtw", len(tw.tw_states))
    print("Qtw0", len(tw.tw_initial_states))
    print("Etw", len(tw.tw_events))
    # print("Etw",tw.tw_events)
    print("ftw", len(tw.tw_trans))

    # [('3', '0', '3', '0'), ('0', '3', '0', '3'), ('2', '1', '2', '1'), ('1', '2', '1', '2')]

    # print("ftw", tw.tw_trans[('2', '1', '3', '4')])
    # print("ftw", tw.tw_trans[('3', '4', '3', '4')])

    # print("state", tw.tw_states)
    # print("innitial", tw.tw_initial_states)
    # print("event", tw.tw_events)
    # print("transfunction", tw.tw_trans)

    reach = Reach(tw.tw_states, tw.tw_initial_states, tw.tw_trans, t_inputo, t_inputu, k1, k2)
    reach.start()

    end = time.perf_counter()

    print("running time", end-start,"s")

'''#文献中的例子
    # states, input_symbols, trans, start_state, final_states
    t_states = ["0","1", "2", "3","4"]
    t_input = ["a", "b", "u"]
    t_inputo = ["a", "b"]
    t_inputu = ["u"]
    t_trans = {"0": {"a":[ "4"],
                     "u": ["2"]},
               "1": {"a": ["4"]},
               "2": {"a": ["3"]},
               "3": {"b": ["3"]},
               "4": {"b": ["4"]}
               }
    t_start = ["1","0"]
'''
'''
    # states, input_symbols, trans, start_state, final_states
    t_states = ["0","1", "2", "3","4","5","6","7","8"]
    t_input = ["up", "d", "r","l"]
    t_inputo = [ "d","r","l"]
    t_inputu = ["up"]
    t_trans = {"0": {"up":["3"],
                     "r": ["1"]},
               "1": {"up":["4"],
                     "l": ["0"],
                     "r": ["2"]},
               "2": {"up": ["5"],
                     "l": ["1"]},
               "3": {"up": ["6"],
                     "r": ["4"],
                     "d": ["0"]},
               "4": {"up":["7"],
                     "d": ["1"],
                     "l": ["3"],
                     "r": ["5"]},
               "5": {"up":["8"],
                     "d": ["2"],
                     "l": ["4"]},
               "6": {"d": ["3"],
                     "r": ["7"]},
               "7": {
                     "d": ["4"],
                     "l": ["6"],
                     "r": ["8"]},
               "8": {
                     "d": ["5"],
                     "l": ["7"]}
               }
    t_start = ["0"]  # 先不管自动机中的初始状态
'''
'''
    # states, input_symbols, trans, start_state, final_states
    t_states = ["0","1", "2", "3","4","5","6","7","8"]
    t_input = ["up", "d", "r","l"]
    t_inputo = [ "d","r","l"]
    t_inputu = ["up"]
    t_trans = {"0": {"up":["3"],
                     "r": ["1"]},
               "1": {"up":["4"],
                     "l": ["0"],
                     "r": ["2"]},
               "2": {"up": ["5"],
                     "l": ["1"]},
               "3": {"up": ["6"],
                     "r": ["4"],
                     "d": ["0"]},
               "4": {"up":["7"],
                     "d": ["1"],
                     "l": ["3"],
                     "r": ["5"]},
               "5": {"up":["8"],
                     "d": ["2"],
                     "l": ["4"]},
               "6": {"d": ["3"],
                     "r": ["7"]},
               "7": {
                     "d": ["4"],
                     "l": ["6"],
                     "r": ["8"]},
               "8": {
                     "d": ["5"],
                     "l": ["7"]}
               }
    t_start = ["0"]  # 先不管自动机中的初始状态
'''