from app.api.dependencies import CurrentUser
from app.modules.frameworks.generator import PillarGenerator
from app.modules.frameworks.repository import PillarContextRepository
from app.modules.frameworks.schemas import (
    GeneratePillarsRequest,
    GeneratePillarsResponse,
)


class PillarGenerationService:
    def __init__(
        self,
        generator: PillarGenerator,
        repository: PillarContextRepository,
    ) -> None:
        self._generator = generator
        self._repository = repository

    async def generate(
        self, request: GeneratePillarsRequest, user: CurrentUser
    ) -> GeneratePillarsResponse:
        await self._repository.mark_running(request)
        try:
            context = await self._repository.load(request, user)
            result = await self._generator.generate(context)
            serialized_result = result.model_dump(mode="json")
            serialized_result["inputs_summary"] = context.model_dump(
                mode="json", exclude={"prompt": {"template"}}
            )
            await self._repository.persist(
                request,
                str(context.approach_id),
                serialized_result,
            )
            return result
        except Exception as error:
            try:
                await self._repository.mark_failed(request, str(error))
            except Exception:
                pass
            raise
