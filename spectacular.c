//该计算器简单的连加减，连乘除，不考虑优先级
#include<stdio.h>
#include<string.h>
#include <stdlib.h>
#include <malloc.h>
#define LINE 2014
#define STACKLENGTH 1024
int top_operate=-1,top_num=-1;
char stack_operate[STACKLENGTH];        //set a stack
char stack_num[STACKLENGTH];           //set a stack

// push a character into the stack
void stack_push(char c,int flag){   //flag=0,operator; flag=1,number
    if(flag==0){
        top_operate = top_operate + 1;         //this is uesed to puch a character
        stack_operate[top_operate] = c;}
    if(flag==1){
        top_num = top_num + 1;         //this is uesed to puch a character
        stack_num[top_num] = c;}
}

int stack_top(char *c,int flag){     //flag=0,operator; flag=1,number
    if(flag==0){
      if( top_operate == -1)
           return 1;             //if the stack is empty return 1
      else{
          *c=stack_operate[top_operate];           //overites the character pointed by c with the caracter currrently on top of stack
          return 0;}
    }
    if(flag==1){
       if( top_num == -1)
          return 1;             //if the stack is empty return 1
      else{
         *c=stack_num[top_num];           //overites the character pointed by c with the caracter currrently on top of stack
         return 0;}
    }

}

int stack_pop(char *c,int flag){    //flag=0,operator; flag=1,number
   int re;
   re=stack_top(c,flag);                         //use top function to ovewrite
   if(flag==0){
       if( re == 0 ){
            top_operate--;                        //if top function overwirte successfully, than remove the pointer into last character
            return 0;}
       if( re == 1 )
            return 1;                     //if top function failed which the stack is empty, than return 1
   }
   if(flag==1){
       if( re == 0 ){
            top_num--;                        //if top function overwirte successfully, than remove the pointer into last character
            return 0;}
       if( re == 1 )
            return 1;                     //if top function failed which the stack is empty, than return 1
   }

}

int main() {    //flag=0,operator; flag=1,number               //initialize an unknown sized string
   char formula[100];
   printf( "Enter a fomula :");
   scanf("%s", formula);
   for(int i=0;formula[i]!=NULL;i++){
       if(formula[i]!= '+'&&formula[i]!= '-'&&formula[i]!= '*'&&formula[i]!= '/')
           stack_push(formula[i],1);
       else
           stack_push(formula[i],0);
   }
   printf( "\ntop_operate: %d ", top_operate);
   printf( "\top_num: %d ", top_num);
   char *ch1;
   ch1=(char*)malloc(LINE*sizeof(char));         //set a dynamic variable use malloc()
   char *ch2;
   ch2=(char*)malloc(LINE*sizeof(char));         //set a dynamic variable use malloc()
   int m=0,total=0;
   m =stack_pop(ch1,1);
   total=atoi(ch1);
   ch1++;
   while(top_num>=0){  //flag=0,operator; flag=1,number
       m=stack_pop(ch1,1);
       m=stack_pop(ch2,0);
        if(*ch2=='+'){
            total=total+atoi(ch1);}
        if(*ch2=='-'){
            if(top_num==-1)
                total=atoi(ch1)-total;
            else
                total=total+atoi(ch1);}
        if(*ch2=='*')
            total=total*atoi(ch1);
        if(*ch2=='/'){
            if(top_num==-1)
                total=total/atoi(ch1);
            else
                total=total*atoi(ch1);}
        ch1++;
        ch2++;
   }
   printf("total:%d",total);
   printf( "\noperator: %s ", stack_operate);
   printf( "\nnumber: %s ", stack_num);

   return 0;
}
