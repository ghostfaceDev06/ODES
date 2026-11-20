
clc
clear
close all
% Parâmetros iniciais
t0 = 0;
tf = 1;
h = 0.2;
t = t0:h:tf;
n = length(t);

% Função EDO: y' = f(t, y)
f = @(t, y) 2*y^2*t^2;

% Solução exata para comparação
y_exata = @(t) 1./((-2./3.)*t.^3+1.);

% Inicialização dos vetores
y_euler = zeros(1, n);

y_euler(1) = 1;

% Implementação do Método de Euler
for i = 1:n-1
    % Método de Euler Explícito
    f_euler = f(t(i), y_euler(i));
    y_euler(i+1) = y_euler(i) + h * f_euler;
    
end

% Gráfico comparativo
figure;
plot(t, y_exata(t), 'k-', 'LineWidth', 2); hold on;
plot(t, y_euler, 'ro--', 'LineWidth', 1.5);
xlabel('Tempo (t)');
ylabel('Valor de y');
legend('Exata', 'Euler', 'Location', 'Best');
title('Comparação: Método de Euler vs Solução Exata');
grid on;

u = y_exata(t)
erro_absoluto=abs(y_euler - u)


 for i = 1:n
   erro_percentual(i)=abs(erro_absoluto(i)/y_euler(i))*100;
 end
erro_percentual 
