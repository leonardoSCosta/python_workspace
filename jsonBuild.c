#include<stdlib.h>
#include<stdio.h>
#include<string.h>

typedef struct parameter
{
    char* name;
}parameter_t;

void add_json_object(char name[], parameter_t* names, float* values, int n_params,
                char jsonString[], short int last_data, short int first_data)
{
    if(first_data)
    {
        sprintf(jsonString, "");
        jsonString = strcat(jsonString, "Hora: epoch/hh:mm:ss\n");
        jsonString = strcat(jsonString, "Data: dd/mm/aaaa\n");
    }

    char comma[3];

    sprintf(comma, "");
    if(!last_data)
        sprintf(comma, ",\n");

    jsonString = strcat(jsonString, "\"");
    jsonString = strcat(jsonString, name);
    jsonString = strcat(jsonString, "\": {");

    int i;
    char aux[10];
    for(i = 0; i < n_params; ++i)
    {
        gcvt(values[i], 3, aux);

        jsonString = strcat(jsonString, "\"");
        jsonString = strcat(jsonString, names[i].name);
        jsonString = strcat(jsonString, "\": ");
        jsonString = strcat(jsonString, aux);

        if( i == n_params - 1)
            jsonString = strcat(jsonString, "}");
        else
            jsonString = strcat(jsonString, ", ");
    }
    jsonString = strcat(jsonString, comma);

}

void main()
{
    char* object_names[] = {"A", "B", "C"};
    parameter_t parameter_names[] = { "v", "a", "fp", "w", "var", "va", "f", "qe"};
    float values[] = {110.5, 0.55, 0.9, 1.01, 0.11, 0.01, 30.0, 1};

    char jsonString[400];

    add_json_object(object_names[0], parameter_names, values, 8, jsonString, 0, 1);
    add_json_object(object_names[1], parameter_names, values, 8, jsonString, 0, 0);
    add_json_object(object_names[2], parameter_names, values, 8, jsonString, 1, 0);
    puts(jsonString);

    printf("Ok");
}
