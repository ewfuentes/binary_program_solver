using JSON
data = []
for i in readdir("src/solved")
    if endswith(i, ".json")
        println(i)
        x = JSON.parsefile("src/solved/$i")
        push!(data, x)
    end
end
open("compact.json", "w") do f
    JSON.print(f, data)
end