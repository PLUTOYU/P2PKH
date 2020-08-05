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
   int total=stack_num[top_num];
   top_num=top_num-1;
   int num1=0;
   char operate;

   char b[10]="qweryuiop";
   for(int i=0;i<10;i++){
    printf("%s 1 \n",b[i]);
   }
   //printf("bbbbbb%s",b[2]);

   while(top_num>=0){
       num1=stack_num[top_num];
       operate=stack_operate[top_operate];
       printf("1111111");
       printf("operate %d",num1);
        if(operate=='+')
            total=total+num1;
        if(operate=='-')
            total=total-num1;
        if(operate=='*')
            total=total*num1;
        if(operate=='/')
            total=total/num1;
        top_num--;
        top_operate--;
   }

   printf( "\total: %d ", total);
   printf( "\noperator: %s ", stack_operate);
   printf( "\nnumber: %s ", stack_num);
   printf("\n");



   return 0;
}
/*
   while(top_num>=0){
        num=stack_pop(stack_num,1);
        operate=stack_pop(stack_operate,0);
        if(cun_op=='+')
            total=total+cun_num;
        if(cun_op=='-')
            total=total-cun_num;
        if(cun_op=='*')
            total=total*cun_num;
        if(cun_op=='/')
            total=total/cun_num;
   }*/
