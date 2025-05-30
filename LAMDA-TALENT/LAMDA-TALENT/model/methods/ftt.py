from model.methods.base import Method

class FTTMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'indices')

    def construct_model(self, model_config = None):
        from model.models.ftt import Transformer
        if model_config is None:
            model_config = self.args.config['model']
        self.model = Transformer(
                d_numerical=self.d_in,
                categories=self.categories,
                d_out=self.d_out,
                **model_config
                ).to(self.args.device) 
        self.model.double()

    