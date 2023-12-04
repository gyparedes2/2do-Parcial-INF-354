CREATE DATABASE EJER5_354
USE EJER5_354
declare @cadena1 varchar(20),
		@cadena2 varchar(20),
		@longitud_cadena1 int,
		@contador int,
		@letra varchar(20),
		@sql nvarchar(4000),
		@columna nvarchar(10)

set @cadena1='gustavo'
set @contador=1
select @longitud_cadena1=LEN(@cadena1) 
set @sql='create table nombre ('
while @contador<=@longitud_cadena1
begin
  set @letra=LEFT(@cadena1, 1)+cast(@contador as varchar(1)) + ' int,'
  set @cadena1=RIGHT(@cadena1, LEN(@cadena1)-1)
  set @sql = @sql + @letra
  set @contador=@contador+1
end
set @sql=LEFT(@sql, LEN(@sql)-1)
set @sql=@sql+')'
--print @sql
exec sp_executesql @sql

set @cadena1='gustao'
set @contador=1
select @longitud_cadena1=LEN(@cadena1) 
while @contador<=@longitud_cadena1
begin
  set @letra=LEFT(@cadena1, 1)
  set @cadena1=RIGHT(@cadena1, LEN(@cadena1)-1)
  set @columna=(select top 1 COLUMN_NAME
		from INFORMATION_SCHEMA.COLUMNS
		where TABLE_NAME='nombre'
		and left(COLUMN_NAME,1)=@letra
		and ORDINAL_POSITION>=@contador)
  set @sql = 'insert into nombre('+@columna+') values(1)'
  --print @sql
  exec sp_executesql @sql
  set @contador=@contador+1
end


declare @consulta nvarchar(4000),
		@consulta2 nvarchar(4000),
		@consulta3 nvarchar(4000),
		@i int,
		@var int,
		@suma_columna int,
		@suma int,
		@nombre_columna varchar(50)
set @consulta = ''
set @consulta2 = ''
set @consulta3 = ''
set @var = (select COUNT(COLUMN_NAME)
from INFORMATION_SCHEMA.COLUMNS
where TABLE_NAME = 'nombre')

set @i = 1
while @i <= @var
begin
	print @i
	set @nombre_columna = (select COLUMN_NAME
	from INFORMATION_SCHEMA.COLUMNS
	where TABLE_NAME = 'nombre'
	and ORDINAL_POSITION = @i)
	set @consulta = @consulta + 'COALESCE(SUM(' + @nombre_columna + '),0)+'
	set @i = @i + 1;
	print @consulta
end
print @consulta
set @consulta2 = LEFT(@consulta,LEN(@consulta)-1)
print @consulta2
set @consulta3 = 'select ' + @consulta2 + ' from nombre'
print @consulta3
exec sp_executesql @consulta3

select * from nombre

--select COALESCE(SUM(g1),0)+COALESCE(SUM(u2),0)+COALESCE(SUM(s3),0)+COALESCE(SUM(t4),0)+COALESCE(SUM(a5),0)+COALESCE(SUM(v6),0)+COALESCE(SUM(o7),0) from nombre

drop table nombre

