
clc
clear
close all
% Parâmetros iniciais
t0 = 1;
tf = 1.5;
h = 0.5;
t = t0:h:tf;
n = length(t);

% Função EDO: y' = f(t, y)
f = @(t, y) 2*t*y;

% Solução exata para comparação
y_exata = @(t) exp(t.^2-1);

% Inicialização dos vetores
y_RK4 = zeros(1, n);


y_RK4(1) = 1;

% Implementação do Método de Runge Kutta de Quarta Ordem 
for i = 1:n-1
    % Método de Runge Kutta
    f_1 = h*f(t(i), y_RK4(i));
    f_2 = h*f(t(i)+(1/2)*h, y_RK4(i)+(1/2)*f_1);
    f_3 = h*f(t(i)+(1/2)*h, y_RK4(i)+(1/2)*f_2);
    f_4 = h*f(t(i) + h,y_RK4(i)+f_3)
    y_RK4(i+1) = y_RK4(i) + (1/6) * (f_1 + 2*f_2 + 2*f_3 + f_4);
    
end

% Inicialização dos vetores
y_euler = zeros(1, n);
y_modif = zeros(1, n);

y_euler(1) = 1;
y_modif(1) = 1;

% Implementação do Método de Euler
for i = 1:n-1
    % Método de Euler Explícito
    f_euler = f(t(i), y_euler(i));
    y_euler(i+1) = y_euler(i) + h * f_euler;
    
end


% Implementação do Método de Euler Modificado
for i = 1:n-1
    % Método de Euler Explícito
    f_euler1 = f(t(i), y_modif(i));
    f_euler2 = f(t(i)+h, y_modif(i)+h*f_euler1);
    y_modif(i+1) = y_modif(i) + (h/2) * (f_euler1+f_euler2);
    
end






% Gráfico comparativo
figure;
plot(t, y_exata(t), 'k-', 'LineWidth', 2); hold on;
plot(t, y_RK4, 'ro--', 'LineWidth', 1.5);
plot(t, y_modif, 'g-', 'LineWidth', 2); hold on;
plot(t, y_euler, 'bo--', 'LineWidth', 1.5);
xlabel('Tempo (t)');
ylabel('Valor de y');
legend('Exata', 'RK_4','Euler_Modif', 'Euler' , 'Location', 'Best');
title('Comparação: Solução Exata vs Runge-Kutta de Quarta Ordem vs Euler e Heun');
grid on;

u = y_exata(t)
erro_absoluto=abs(y_RK4 - u)


for i = 1:n
   erro_percentual(i)=abs(erro_absoluto(i)/y_RK4(i))*100;
end
erro_percentual


format short  %long

A = [y_euler'  y_modif' y_RK4' u' erro_percentual'];

T = array2table(A,VariableNames=["Euler" "Euler Modif" "RK4" "Exata" "Erro Percentual"])
